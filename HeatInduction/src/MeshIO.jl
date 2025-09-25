module MeshIO
include("Structs.jl")
using LinearAlgebra
using .Structs


function readOpenFoamMesh(caseDir::String)
    if !isdir(caseDir)
        throw(CaseDirError("Case Directory '$(caseDir)' does not exist"))
    end
    polymeshDir = joinpath(caseDir, "constant/polyMesh")
    if !isdir(polymeshDir)
        throw(CaseDirError("PolyMesh Directory '$(caseDir)' does not exist"))
    end
    println("Reading OpenFOAM mesh files from mesh directory: $caseDir")


    nodes = readPointsFile(polymeshDir)
    owner = readOwnersFile(polymeshDir)
    neighbors = readNeighborsFile(polymeshDir)
    faces = readFacesFile(polymeshDir, owner, neighbors)
    numCells = maximum(owner) + 1
    boundaries = readBoundaryFile(polymeshDir)
    mesh = constructCells(nodes, boundaries, faces, numCells, neighbors)
    mesh = setupNodeConnectivities(mesh)
    mesh = MeshProcessor.processOpenFoamMesh(mesh)
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
        push!(nodes, Node(coords, [], [], [], 0))
    end
    return nodes

end # function readPointsFile

function readFacesFile(polyMeshDir::String, owners::Vector{Int}, neighbors::Vector{Int})::Vector{Face}
    facesFileName = joinpath(polyMeshDir, "faces")
    if !isfile(facesFileName)
        throw(CaseDirError("Faces file '$(facesFileName)' does not exist."))
    end
    lines = readlines(facesFileName)[22:(end-4)]
    faces::Vector{Face} = []
    for (index, line) in enumerate(lines)
        s = split((line[3:(end-1)]), " ")
        nodes = map(s -> tryparse(Int, s) + 1, s)
        push!(faces, Face(index, nodes, owners[index], 0, [], [], 0.0, [], 0.0, [], 0.0, [], 0.0, 0))
    end
    for (i, nb) in enumerate(neighbors)
        faces[i].iNeighbor = nb
    end
    return faces
end # function readPointsFile

function readOwnersFile(polyMeshDir::String)::Vector{Int}
    ownersFileName = joinpath(polyMeshDir, "owner")
    if !isfile(ownersFileName)
        throw(CaseDirError("Owners file '$(ownersFileName)' does not exist."))
    end
    lines = readlines(ownersFileName)[23:(end-4)]
    owners = map(l -> tryparse(Int, l)+1, lines)
    return owners
end # function readPointsFile


function readNeighborsFile(polyMeshDir::String)::Vector{Int}
    neighborsFileName = joinpath(polyMeshDir, "neighbour")
    if !isfile(neighborsFileName)
        throw(CaseDirError("Neighbors file '$(neighborsFileName)' does not exist."))
    end
    line = readlines(neighborsFileName)[20]
    s = split((line[3:(end-1)]), " ")
    nbs = map(s -> tryparse(Int, s), s)
    return nbs
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

function constructCells(nodes::Vector{Node}, boundaries::Vector{Boundary}, faces::Vector{Face}, numCells::Int, neighbors::Vector{Int})::Structs.Mesh
    numInteriorFaces = size(neighbors)[1]
    cells = [Cell(index, [], [], 0, [], [], 0.0, 0.0, []) for index in 1:numCells]
    for interiorFace::Int in 1:numInteriorFaces
        owner = faces[interiorFace].iOwner + 1
        nb = neighbors[interiorFace]
        push!(cells[owner].iFaces, nb)
        push!(cells[owner].neighbors, nb)
        push!(cells[owner].faceSigns, 1)

        push!(cells[nb].iFaces, faces[interiorFace].index)
        push!(cells[nb].neighbors, owner)
        push!(cells[nb].faceSigns, -1)
    end
    for boundaryFace in numInteriorFaces:size(faces)[1]
        owner = faces[boundaryFace].iOwner + 1
        push!(cells[owner].iFaces, boundaryFace)
        push!(cells[owner].faceSigns, 1)
    end

    for cell in cells
        cell.numNeighbors = size(cell.neighbors)[1]
    end
    numBoundaryCells = size(faces)[1] - numInteriorFaces
    numBoundaryFaces = size(faces)[1] - numInteriorFaces
    return Structs.Mesh(nodes, faces, boundaries, numCells, cells, numInteriorFaces, numBoundaryCells, numBoundaryFaces)
end # function constructCells

function setupNodeConnectivities(mesh::Mesh)::Structs.Mesh
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


module MeshProcessor
using ..Structs
using ..LinearAlgebra

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
    println("Processing Mesh Geometry")
    for (index, face) in enumerate(mesh.faces)
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
            for (index, iNode) in enumerate(face.iNodes)
                triangleNode2 = mesh.nodes[iNode].centroid
                if index < size(face.iNodes)[1] - 1
                    triangleNode3 = mesh.nodes[face.iNodes[index+1]].centroid
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
    println("Computing Element Volumen and Centroid")
    for cell in mesh.cells
        elementCenter = zeros(3)
        elementCenter = sum(map(f -> f.centroid, mesh.faces[cell.iFaces])) / size(cell.iFaces)[1]

        # Compute volume and centroid of each element
        elementCentroid = zeros(3)
        localVolumeCentroidSum = zeros(3)
        localVolumeSum = 0.0
        for i in 1:size(cell.iFaces)[1]
            localFace = mesh.faces[i]
            localFaceSign = cell.faceSigns[i]
            Sf = localFaceSign * localFace.Sf

            # Calculate the distance vector from geometric center to the face centroid
            d_Gf = localFace.centroid - elementCenter

            # Calculate the volume of each sub-element pyramid
            localVolume = dot(Sf, d_Gf) / 3.0
            localVolumeSum += localVolume

            #Calculate volume-weighted center of sub-element pyramid (centroid)
            localCentroid = localFace.centroid * 0.75 + 0.25 * elementCenter
            localVolumeCentroidSum += localCentroid * localVolume
        end
        cell.centroid = (1 / localVolumeSum)' * localVolumeCentroidSum
        cell.volume = localVolumeSum
        cell.oldVolume = localVolumeSum
    end
    return mesh
end # function computeElementVolumeAndCentroid

function processSecondaryFaceGeometry(mesh::Mesh)::Mesh
    # Loop over interior faces
    for face in mesh.faces[1:mesh.numInteriorFaces]
        # Compute unit surface normal vector
        nf = (1 / face.area) * face.Sf
        ownerElement = mesh.cells[face.iOwner]
        neighborElement = mesh.cells[face.iNeighbor]
        face.CN = neighborElement.centroid - ownerElement.centroid

        face.magCN = magnitude(face.CN)
        face.eCN = (1 / face.magCN) * face.CN

        E = face.area * face.eCN

        face.gDiff = magnitude(E) / face.magCN

        face.T = face.Sf - E

        # Compute face weighting factor
        Cf = face.centroid - ownerElement.centroid
        fF = neighborElement.centroid - face.centroid
        face.gf = dot(Cf, nf) / (dot(Cf, nf) + dot(fF, nf))
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
    for (index, boundary) in enumerate(mesh.boundaries)
        startFace = boundary.startFace
        nBFaces = boundary.nFaces
        for face in mesh.faces[startFace:(startFace+nBFaces)]
            face.patchIndex = index
        end
    end
    return mesh
end # function labelBoundaryFaces

end # module MeshProcessor

end # module MeshIO
