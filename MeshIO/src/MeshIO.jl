module MeshIO
using LinearAlgebra
using MeshStructs
using MeshProcessor

function readOpenFoamMesh(caseDir::String)::Mesh
    if !isdir(caseDir)
        throw(CaseDirError("Case Directory '$(caseDir)' does not exist"))
    end
    polymeshDir = joinpath(caseDir, "constant/polyMesh")
    if !isdir(polymeshDir)
        throw(CaseDirError("PolyMesh Directory '$(caseDir)' does not exist"))
    end
    println("Reading OpenFOAM mesh files from mesh directory: $caseDir")

    nodes = readPointsFile(polymeshDir)
    println(nodes)
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
    for (index, line) in enumerate(lines[22:(end-4)])
        s = split((line[3:(end-1)]), " ")
        nodes = map(s -> tryparse(Int, s) + 1, s)
        println("face index: $(index), nnodes: $(size(nodes))")
        push!(faces, Face(index, nodes, owners[index], -1, zeros(3), zeros(3), 0.0, zeros(3), 0.0, zeros(3), 0.0, zeros(3), 0.0, -1))
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
    line = readlines(neighborsFileName)[20]
    println("line: $(line[1])")
    numNeighbors = tryparse(Int, "$(line[1])")
    s = split((line[3:(end-1)]), " ")
    nbs = map(s -> tryparse(Int, s), s)
    println("nbs: $(size(nbs))")
    for iNeighbor in 1:numNeighbors
        println("adding nb $(nbs[iNeighbor]) to face $iNeighbor")
        faces[iNeighbor].iNeighbor = nbs[iNeighbor]
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
        println("pushing back $(iNeighbor) for cell $(iOwner)")
        push!(cells[iOwner].iNeighbors, iNeighbor)
        push!(cells[iOwner].faceSigns, 1)

        push!(cells[iNeighbor].iFaces, faces[interiorFace].index)
        push!(cells[iNeighbor].iNeighbors, iOwner)
        println("pushing back $(iOwner) for cell $(iNeighbor)")

        push!(cells[iNeighbor].faceSigns, -1)
    end
    for boundaryFace in numInteriorFaces:size(faces)[1]
        owner = faces[boundaryFace].iOwner
        println("pushing back b $(boundaryFace) for cell $(owner)")
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

export readOpenFoamMesh
end # module MeshIO