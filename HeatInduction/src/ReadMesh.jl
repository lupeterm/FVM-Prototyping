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
    i = 18
    lines = readlines(pointsFileName)
    while isempty(lines[i])
        i += 1
    end
    nNodes = tryparse(Int, lines[i])
    i += 2
    nodes::Vector{Node} = []
    for line in lines[i:i+nNodes-1]
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
    i = 18
    lines = readlines(facesFileName)
    while isempty(lines[i])
        i += 1
    end
    nfaces = tryparse(Int, lines[i])
    faces::Vector{Face} = []
    i += 2
    for (index, iFace) in enumerate(i:i+nfaces-1)
        s = split((lines[iFace][3:(end-1)]), " ")
        nodes = map(s -> tryparse(Int, s) + 1, s)
        push!(faces, Face(index, nodes, owners[index], -1, zeros(3), zeros(3), 0.0, zeros(3), 0.0, zeros(3), 0.0, zeros(3), 0.0, -1, 0.0, 0, 0))
    end
    return faces
end # function readPointsFile

function readOwnersFile(polyMeshDir::String)::Vector{Int}
    ownersFileName = joinpath(polyMeshDir, "owner")
    if !isfile(ownersFileName)
        throw(CaseDirError("Owners file '$(ownersFileName)' does not exist."))
    end
    i = 18
    lines = readlines(ownersFileName)
    while isempty(lines[i]) || startswith(lines[i], "//")
        i += 1
    end
    nOwners = tryparse(Int, lines[i])
    i += 2
    owners = []
    for iOwner in i:i+nOwners-1
        owner = tryparse(Int, lines[iOwner]) +1
        push!(owners, owner)
    end
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
    i = 17
    lines = readlines(boundariesFileName)
    while isempty(lines[i]) || startswith(lines[i], "//")
        i += 1
    end
    boundaries::Vector{Boundary} = []
    lines = readlines(boundariesFileName)[i+2:(end-4)]
    l = split(join(lines), "}")
    for bound in l
        s = replace(bound, r"\s+" => " ")
        s = replace(bound, ";" => "")
        s = split(s, " ")
        s = filter(x -> x != "", s)
        if s[5] == "inGroups" 
            ing1 = tryparse(Int, "$(s[6][1])")
            ingroups = (ing1, s[6][3:(end-1)])
            boundary = Boundary(
                s[1],
                s[4],
                ingroups,
                tryparse(Int, s[8]),
                tryparse(Int, s[10]),
            )
        else
            boundary = Boundary(
                s[1],
                s[4],
                (-1, "null"),
                tryparse(Int, s[6]),
                tryparse(Int, s[8]),
            )
        end

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
    for iFace in 1:size(mesh.faces)[1]
        iNodes = mesh.faces[iFace].iNodes
        nNodes = size(iNodes)[1]
        for iNode in 1:nNodes
            push!(mesh.nodes[iNodes[iNode]].iFaces, mesh.faces[iFace].index)
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
        face.centroid = centroid
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
        localVolumeCentroidSum = zeros(3)
        localVolumeSum = 0.0
        for iFace in 1:size(iFaces)[1]
            localFace = mesh.faces[iFaces[iFace]]
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
        
        # vector between cell centroids
        theFace.CN = neighborElement.centroid - ownerElement.centroid
        # length of said vector / direct distance between cell centroids
        theFace.magCN = magnitude(theFace.CN)
        # normalized vector to neighbor
        theFace.eCN = (1 / theFace.magCN) * theFace.CN

        E = theFace.area * theFace.eCN
        # essentially strength of flux in the direction of neighboring cell
        # eq 8.11: gdiff of face = ||face Surface || / || distance between cell centroids ||
        theFace.gDiff = magnitude(E) / magnitude(theFace.CN)
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

function readFields(dir::String, mesh::Mesh)::Tuple{Vector{Vector{BoundaryField}}, Dict{Int, String}}
    files = readdir(dir)
    fields::Vector{Vector{BoundaryField}} = []
    mappings = Dict()
    for (i, variableFile) in enumerate(files)
        fileName = joinpath(dir, variableFile)
        bfields = readField(fileName, mesh)
        mappings[i] = variableFile
        push!(fields, bfields) 
    end
    return fields, mappings
end # function readTemperatureField

"""
    parse '(0 0 0)' to [0.0, 0.0, 0.0]
"""
parseVec(string::String)::Vector{Float64} = map(s -> tryparse(Float64,s ), split(string[2:end-1], " "))
parseVec(sub::SubString)::Vector{Float64} = parseVec(String(sub))

function readField(filePath::String, mesh::Mesh)::Vector{BoundaryField}
    fileName = rsplit(filePath, "/", limit=1)[1]
    if !isfile(filePath)
        throw(CaseDirError("Field file '$(filePath)' does not exist."))
    end
    i = 17
    lines = readlines(filePath)
    while !startswith(lines[i], "internalField")
        i += 1
    end
    split18 = match(r"^(\w+)\s+(\w+)\s(?:(\d)|(\([\d\s]+\)));", lines[i])
    # TODO so far not used 
    numCells = size(mesh.cells)[1]
    internalValues = []
    if split18[1] == "internalField" && split18[2] == "uniform"
        if isnothing(split18[3])
            # parse '(0 0 0)' to [0.0, 0.0, 0.0]
            value = parseVec(split18[4])
        else
            value = tryparse(Int, split18[3])
        end
        internalValues = fill(value, numCells)
    end
    internalField = Field(numCells, internalValues)
    while !contains(lines[i], "boundaryField")
        i += 1
    end
    boundaryLines = lines[i+2:end-4]
    joined = join(boundaryLines)
    splitted = split(joined, "}")
    boundaryFields = []
    for boundary in mesh.boundaries
        for b in splitted
            matches = match(r"\s*(\w+)\s*\{\s*type\s*(\w+);(?:\s*value\s*(\w+) (?:(\d+)|(\([\d\s]+\)));)?", b)
            values = []
            if boundary.name != matches[1]
                continue
            end
            nFaces = boundary.nFaces
            type = matches[2]
            if matches[2] == "empty"
                values = []
                nFaces = 0
            end
            if !isnothing(matches[3])
                if !isnothing(matches[4])  # scalars like temperature
                    scalar = tryparse(Float64, matches[4])
                    values = fill(scalar, nFaces) 
                else  # vectors like velocity
                    vector = parseVec(matches[5])
                    values = fill(vector, nFaces) 
                end
            end
            bfield = BoundaryField(String(fileName), nFaces, values, String(type))
            push!(boundaryFields, bfield)
        end
    end
    return boundaryFields
end

function readPropertiesFile(path::String)::Float64
    if !isfile(path)
        throw(CaseDirError("Field file '$(path)' does not exist."))
    end
    file = read(path, String)
    variable = match(r"\w+\s*\[[\s\d-]+\]\s*([\.\d]+);", file)
    val = tryparse(Float64, variable[1])
    return val
end