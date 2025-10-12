using LinearAlgebra
using ..MeshStructs

function readOpenFoamMesh(caseDir::String)::Mesh
    if !isdir(caseDir)
        throw(CaseDirError("Case Directory '$(caseDir)' does not exist"))
    end
    polymeshDir = joinpath(caseDir, "constant/polyMesh")
    if !isdir(polymeshDir)
        throw(CaseDirError("PolyMesh Directory '$(caseDir)' does not exist"))
    end
    nodes = readPointsFile(polymeshDir)
    owner = readOwnersFile(polymeshDir)
    faces = readFacesFile(polymeshDir, owner)
    neighbors, faces = readNeighborsFile(polymeshDir, faces)
    numCells = maximum(owner)
    boundaries = readBoundaryFile(polymeshDir)

    mesh = constructCells(nodes, boundaries, faces, numCells, neighbors)
    mesh = setupNodeConnectivities(mesh)
    mesh = processOpenFoamMesh(mesh)
    return mesh
end # function readOpenFoamMesh

function readPointsFile(polyMeshDir::String)::Vector{Node}
    pointsFileName = joinpath(polyMeshDir, "points")
    if !isfile(pointsFileName)
        throw(CaseDirError("Points file '$(pointsFileName)' does not exist."))
    end
    lines = readlines(pointsFileName)[22:(end-4)]
    nodes::Vector{Node} = []
    for line in lines
        s = split((line[2:(end-1)]), " ")
        coords = map(s -> tryparse(Float64, s), s)
        push!(nodes, Node(coords, [], [], 0))
    end
    return nodes

end # function readPointsFile

function readFacesFile(polyMeshDir::String, owners::Vector{Int})::Vector{Face}
    facesFileName = joinpath(polyMeshDir, "faces")
    if !isfile(facesFileName)
        throw(CaseDirError("Faces file '$(facesFileName)' does not exist."))
    end
    lines = readlines(facesFileName)
    faces::Vector{Face} = []
    nfaces = tryparse(Int, lines[20])
    for iFace in 1:nfaces
        s = split((lines[21+iFace][3:(end-1)]), " ")
        nodes = map(s -> tryparse(Int, s) + 1, s)
        push!(faces, Face(iFace, nodes, owners[iFace], -1, zeros(3), zeros(3), 0.0, zeros(3), 0.0, zeros(3), 0.0, zeros(3), 0.0, -1, 0.0, 0, 0))
    end
    return faces
end # function readPointsFile

function readOwnersFile(polyMeshDir::String)::Vector{Int}
    ownersFileName = joinpath(polyMeshDir, "owner")
    if !isfile(ownersFileName)
        throw(CaseDirError("Owners file '$(ownersFileName)' does not exist."))
    end
    lines = readlines(ownersFileName)[23:(end-4)]
    owners = map(l -> tryparse(Int, l) + 1, lines)
    return owners
end # function readPointsFile


function readNeighborsFile(polyMeshDir::String, faces::Vector{Face})::Tuple{Vector{Int},Vector{Face}}
    neighborsFileName = joinpath(polyMeshDir, "neighbour")
    if !isfile(neighborsFileName)
        throw(CaseDirError("Neighbors file '$(neighborsFileName)' does not exist."))
    end
    i = 19 
    lines = readlines(neighborsFileName)
    while isempty(lines[i])
        i += 1
    end
    nn = match(r"(\d+)(?:\([\d\s]+\))?", lines[i])
    numNeighbors = tryparse(Int, nn[1])
    if numNeighbors <= 4
        s = split((lines[i][3:(end-1)]), " ")
        nbs = map(s -> tryparse(Int, s), s)
    else
        nbs = [tryparse(Int, "$(l)") for l in lines[i+2:i+2+numNeighbors-1]]
    end
    for iNeighbor in 1:numNeighbors
        faces[iNeighbor].iNeighbor = nbs[iNeighbor] + 1
    end
    return nbs, faces
end # function readPointsFile

function readBoundaryFile(polyMeshDir::String)::Vector{Boundary}
    boundariesFileName = joinpath(polyMeshDir, "boundary")
    if !isfile(boundariesFileName)
        throw(CaseDirError("Boundary file '$(boundariesFileName)' does not exist."))
    end

    boundaries::Vector{Boundary} = []
    lines = readlines(boundariesFileName)[21:(end-4)]
    l = split(join(lines), "}")
    for bound in l
        s = replace(bound, r"\s+" => " ")
        s = replace(bound, ";" => "")
        s = split(s, " ")
        s = filter(x -> x != "", s)
        ing1 = tryparse(Int, "$(s[6][1])")
        ing = (ing1, s[6][3:(end-1)])
        boundary = Boundary(
            s[1],
            s[4],
            ing,
            tryparse(Int, s[8]),
            tryparse(Int, s[10]),
        )
        push!(boundaries, boundary)
    end
    return boundaries
end # function readPointsFile

function constructCells(nodes::Vector{Node}, boundaries::Vector{Boundary}, faces::Vector{Face}, numCells::Int, neighbors::Vector{Int})::Mesh
    numInteriorFaces = size(neighbors)[1]
    cells = [Cell(index, [], [], 0, [], [], 0.0, 0.0, []) for index in 1:numCells]
    for interiorFace::Int in 1:numInteriorFaces
        iOwner = faces[interiorFace].iOwner
        iNeighbor = faces[interiorFace].iNeighbor
        push!(cells[iOwner].iFaces, faces[interiorFace].index)
        push!(cells[iOwner].iNeighbors, iNeighbor)
        push!(cells[iOwner].faceSigns, 1)

        push!(cells[iNeighbor].iFaces, faces[interiorFace].index)
        push!(cells[iNeighbor].iNeighbors, iOwner)
        push!(cells[iNeighbor].faceSigns, -1)
    end
    for boundaryFace in (numInteriorFaces+1):size(faces)[1]
        owner = faces[boundaryFace].iOwner
        push!(cells[owner].iFaces, boundaryFace)
        push!(cells[owner].faceSigns, 1)
    end

    for cell in cells
        cell.numNeighbors = size(cell.iNeighbors)[1]
    end
    numBoundaryCells = size(faces)[1] - numInteriorFaces
    numBoundaryFaces = size(faces)[1] - numInteriorFaces
    return Mesh(nodes, faces, boundaries, numCells, cells, numInteriorFaces, numBoundaryCells, numBoundaryFaces)
end # function constructCells

function setupNodeConnectivities(mesh::Mesh)::Mesh
    for face::Face in mesh.faces
        for iNode in face.iNodes
            push!(mesh.nodes[iNode].iFaces, face.index)
        end
    end
    for cell in mesh.cells
        for face in mesh.faces[cell.iFaces]
            for iNode in face.iNodes
                if !(iNode in cell.iNodes)
                    push!(cell.iNodes, iNode)
                    push!(mesh.nodes[iNode].iCells, cell.index)
                end
            end
        end
    end
    return mesh
end



function processOpenFoamMesh(mesh::Mesh)::Mesh
    mesh = processBasicFaceGeometry(mesh)
    mesh = computeElementVolumeAndCentroid(mesh)
    mesh = processSecondaryFaceGeometry(mesh)
    mesh = sortBoundaryNodesFromInteriorNodes(mesh)
    mesh = labelBoundaryFaces(mesh)
    return mesh
end # function processOpenFoamMesh

function magnitude(vector::Vector{Float64})::Float64
    return sqrt(dot(vector, vector))
end # function magnitude

function processBasicFaceGeometry(mesh::Mesh)::Mesh
    for face in mesh.faces
        centroid = zeros(3)
        Sf = zeros(3)
        area = 0.0
        # special case: triangle
        if size(face.iNodes)[1] == 3
            # sum x,y,z and divide by 3
            triangleNodes = map(n -> n.centroid, mesh.nodes[face.iNodes])
            centroid = sum(triangleNodes) / 3
            Sf = 0.5 * cross(triangleNodes[2] - triangleNodes[1], triangleNodes[3] - triangleNodes[1])
            area = magnitude(Sf)
        else # general case, polygon is not a triangle 
            nodes = map(n -> n.centroid, mesh.nodes[face.iNodes])
            center = sum(nodes) / size(nodes)[1]
            # Using the center to compute the area and centroid of virtual
            # triangles based on the center and the face nodes
            triangleNode1 = center
            triangleNode2 = zeros(3)
            triangleNode3 = zeros(3)
            for (iNodeIndex, iNode) in enumerate(face.iNodes)
                triangleNode2 = mesh.nodes[iNode].centroid
                if iNodeIndex < size(face.iNodes)[1]
                    triangleNode3 = mesh.nodes[face.iNodes[iNodeIndex+1]].centroid
                else
                    triangleNode3 = mesh.nodes[face.iNodes[1]].centroid
                end
                # Calculate the centroid of a given subtriangle
                localCentroid = (triangleNode1 + triangleNode2 + triangleNode3) / 3
                # Calculate the surface area vector of a given subtriangle by cross product
                localSf = 0.5 * cross(triangleNode2 - triangleNode1, triangleNode3 - triangleNode2)
                # Calculate the surface area of a given subtriangle
                localArea = sqrt(dot(localSf, localSf))
                centroid += localArea * localCentroid
                Sf += localSf
            end
            area = magnitude(Sf)
            # Compute centroid of the polygon
            centroid /= area
        end
        face.centroid = round.(centroid, digits=2)
        face.Sf = Sf
        face.area = area
    end
    return mesh
end # function processBasicFaceGeometry

function computeElementVolumeAndCentroid(mesh)::Mesh
    for iElement in 1:(size(mesh.cells)[1])
        iFaces = mesh.cells[iElement].iFaces
        elementCenter = zeros(3)
        for iFace in iFaces
            elementCenter += mesh.faces[iFace].centroid
        end
        elementCenter /= size(iFaces)[1]

        elementCentroid = zeros(3)
        localVolumeCentroidSum = zeros(3)
        localVolumeSum = 0.0

        for (iFace, localFace) in enumerate(mesh.faces[iFaces])
            localFaceSign = mesh.cells[iElement].faceSigns[iFace]
            Sf = localFaceSign * localFace.Sf
            d_Gf = localFace.centroid - elementCenter

            localVolume = (Sf[1] * d_Gf[1] + Sf[2] * d_Gf[2] + Sf[3] * d_Gf[3]) / 3.0
            localVolumeSum += localVolume

            localCentroid = 0.75 * localFace.centroid + 0.25 * elementCenter
            localVolumeCentroidSum += localCentroid * localVolume
        end
        mesh.cells[iElement].centroid = (1 / localVolumeSum) * localVolumeCentroidSum
        mesh.cells[iElement].volume = localVolumeSum
        mesh.cells[iElement].oldVolume = localVolumeSum
    end
    return mesh
end # function computeElementVolumeAndCentroid

function processSecondaryFaceGeometry(mesh::Mesh)::Mesh
    # Loop over interior faces
    for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        # Compute unit surtheFace normal vector
        nf = (1 / theFace.area) * theFace.Sf
        ownerElement = mesh.cells[theFace.iOwner]
        neighborElement = mesh.cells[theFace.iNeighbor]
        theFace.CN = neighborElement.centroid - ownerElement.centroid
        theFace.magCN = magnitude(theFace.CN)
        theFace.eCN = (1 / theFace.magCN) * theFace.CN

        E = theFace.area * theFace.eCN
        theFace.gDiff = magnitude(E) / theFace.magCN

        theFace.T = theFace.Sf - E

        # Compute theFace weighting factor
        Cf = theFace.centroid - ownerElement.centroid
        fF = neighborElement.centroid - theFace.centroid
        theFace.gf = dot(Cf, nf) / (dot(Cf, nf) + dot(fF, nf))
    end
    for iBFace in (mesh.numInteriorFaces+1):size(mesh.faces)[1]
        theBFace = mesh.faces[iBFace]
        ownerElement = mesh.cells[theBFace.iOwner]
        CN = theBFace.centroid - ownerElement.centroid
        mesh.faces[iBFace].CN = CN
        mesh.faces[iBFace].gDiff = theBFace.area * theBFace.area / dot(CN, theBFace.Sf)
        magCN = magnitude(CN)
        mesh.faces[iBFace].magCN = magCN
        eCN = (1 / magCN) * CN
        E = theBFace.area * eCN
        mesh.faces[iBFace].T = theBFace.Sf - E
        mesh.faces[iBFace].gf = 1.0
        mesh.faces[iBFace].walldist = dot(CN, theBFace.Sf) / magnitude(theBFace.Sf)
    end
    for iElement in mesh.cells
        iFaces = iElement.iFaces
        iNeighbors = iElement.iNeighbors
        kf = 1
        for i in 1:size(iNeighbors)[1]
            iFace = iFaces[i]
            if mesh.faces[iFace].iOwner == iElement
                mesh.faces[iFace].iOwnerNeighborCoef = kf
            elseif mesh.faces[iFace].iNeighbor == iElement
                mesh.faces[iFace].iNeighborOwnerCoef = kf
            end
            kf += 1
        end
    end
    return mesh
end # function processSecondaryFaceGeometry

function sortBoundaryNodesFromInteriorNodes(mesh::Mesh)::Mesh
    for face in mesh.faces[1:mesh.numInteriorFaces]
        for iNode in face.iNodes
            mesh.nodes[iNode].flag = 1
        end
    end
    for boundary in mesh.boundaries
        startFace = boundary.startFace
        nBFaces = boundary.nFaces
        s1 = boundary.name == "frontAndBack"
        s2 = boundary.name == "frontAndBackPlanes"
        for face in mesh.faces[startFace:(startFace+nBFaces)]
            for iNode in face.iNodes
                mesh.nodes[iNode].flag = (s1 || s2) ? 1 : 0
            end
        end
    end
    return mesh
end # function sortBoundaryNodesFromInteriorNodes

function labelBoundaryFaces(mesh::Mesh)::Mesh
    for (iBoundary, boundary) in enumerate(mesh.boundaries)
        startFace = boundary.startFace + 1
        nBFaces = boundary.nFaces
        for iFace in startFace:(startFace+nBFaces-1)
            mesh.faces[iFace].patchIndex = iBoundary
        end
    end
    return mesh
end # function labelBoundaryFaces