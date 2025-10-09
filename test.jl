module HeatInduction

using LinearAlgebra
struct CaseDirError <: Exception
    message::String
end # struct CaseDirError

mutable struct Face
    index::Int
    iNodes::Vector{Int}
    iOwner::Int
    iNeighbor::Int
    centroid::Vector{Float64}
    Sf::Vector{Float64}
    area::Float64
    CN::Vector{Float64}
    magCN::Float64
    eCN::Vector{Float64}
    gDiff::Float64
    T::Vector{Float64}
    gf::Float64
    patchIndex::Int
    walldist::Float64
    iOwnerNeighborCoef::Int
    iNeighborOwnerCoef::Int
end # struct Face

mutable struct Node
    centroid::Vector{Float64}
    iCells::Vector{Int}
    iFaces::Vector{Int}
    flag::Int
end # struct Node

struct Boundary
    name::String
    type::String
    inGroups::Tuple{Int,String}
    nFaces::Int
    startFace::Int
end # struct Boundary

mutable struct Cell
    index::Int
    iFaces::Vector{Int}
    iNeighbors::Vector{Int}
    numNeighbors::Int
    faceSigns::Vector{Int}
    iNodes::Vector{Int}
    volume::Float64
    oldVolume::Float64
    centroid::Vector{Float64}
end # struct Cell

struct Mesh
    nodes::Vector{Node}
    faces::Vector{Face}
    boundaries::Vector{Boundary}
    numCells::Int
    cells::Vector{Cell}
    numInteriorFaces::Int
    numBoundaryCells::Int
    numBoundaryFaces::Int
end # struct Mesh

mutable struct Field
    nElements::Int
    values::Vector{Float64}
end # struct Field

mutable struct BoundaryField
    nFaces::Int
    values::Vector{Float64}
    type::String
end # struct BoundaryField

import Base: show

function show(io::IO, node::Node)
    printNode(node)
end

function show(io::IO, cell::Cell)
    printCell(cell)
end

function show(io::IO, face::Face)
    printFace(face)
end

function show(io::IO, boundary::Boundary)
    printBoundary(boundary)
end

function show(io::IO, mesh::Mesh)
    printMesh(mesh)
end

function printMesh(mesh::Mesh)
    println("Mesh:")
    println("  Number of Cells: ", mesh.numCells)
    println("  Number of Interior Faces: ", mesh.numInteriorFaces)
    println("  Number of Boundary Cells: ", mesh.numBoundaryCells)
    println("  Number of Boundary Faces: ", mesh.numBoundaryFaces)

    println("  Nodes: [")
    for (i, node) in enumerate(mesh.nodes)
        println("    Node[$i]:")
        println(node)
    end
    println("  ]")

    println("  Faces: [")
    for (i, face) in enumerate(mesh.faces)
        println(face)
    end
    println("  ]")

    println("  Boundaries: [")
    for (i, boundary) in enumerate(mesh.boundaries)
        println(boundary)
    end
    println("  ]")

    println("  Cells: [")
    for (i, cell) in enumerate(mesh.cells)
        println(cell)
    end
    println("  ]")
end


function printFace(face::Face)
    println("\tFace $(face.index) {")
    println("\t\tindex: $(face.index)")
    println("\t\tnNodes: ", size(face.iNodes)[1])
    println("\t\tiNodes: ", face.iNodes)
    println("\t\tiOwner: ", face.iOwner)
    println("\t\tiNeighbor: ", face.iNeighbor)
    println("\t\tcentroid: ", face.centroid)
    println("\t\tSf: ", face.Sf)
    println("\t\tarea: ", face.area)
    println("\t\tmagCN: ", face.magCN)
    println("\t\tCN: ", face.CN)
    println("\t\teCN: ", face.eCN)
    println("\t\tgDiff: ", face.gDiff)
    println("\t\tT: ", face.T)
    println("\t\tgf: ", face.gf)
    println("\t\tpatchIndex: ", face.patchIndex)
    println("\t}")
end

function printBoundary(b::Boundary)
    println("\tBoundary {")
    println("\t\tname: ", b.name)
    println("\t\ttype: ", b.type)
    println("\t\tinGroups: (", b.inGroups[1], ", \"", b.inGroups[2], "\")")
    println("\t\tnFaces: ", b.nFaces)
    println("\t\tstartFace: ", b.startFace)
    println("\t}")
end

function printNode(node::Node)
    println("\tNode:")
    println("\t\tCentroid: ", node.centroid)
    println("\t\tiCells: ", node.iCells)
    println("\t\tiFaces: ", node.iFaces)
    println("\t\tFlag: ", node.flag)
end

function printCell(cell::Cell)
    println("\tCell $(cell.index):")
    println("\t\tiFaces: ", cell.iFaces)
    println("\t\tiNeighbors: ", cell.iNeighbors)
    println("\t\tNumNeighbors: ", cell.numNeighbors)
    println("\t\tFaceSigns: ", cell.faceSigns)
    println("\t\tiNodes: ", cell.iNodes)
    println("\t\tVolume: ", cell.volume)
    println("\t\tOldVolume: ", cell.oldVolume)
    println("\t\tCentroid: ", cell.centroid)
end


function readTemperatureField(caseDir::String, mesh::Mesh, internalTemperatureField::Field)::Vector{BoundaryField}
    TFileName = joinpath(caseDir, "0/T")
    if !isfile(TFileName)
        throw(CaseDirError("T file '$(TFileName)' does not exist."))
    end
    lines = readlines(TFileName)[19:(end)]
    split18 = match(r"(\w+)\s+(\w+)\s(\d+);", lines[1])
    if split18[1] == "internalField" && split18[2] == "uniform"
        value = tryparse(Int, split18[3])
        internalTemperatureField.values = [value for x in internalTemperatureField.values]
    end
    boundaryTemperatureFields = [BoundaryField(0, [], "") for _ in 1:(size(mesh.boundaries)[1])]

    if occursin("boundaryField", lines[3])
        pointer = 5
        for (index, boundary) in enumerate(mesh.boundaries)
            name = strip(lines[pointer])
            println("$(name), $(boundary.name)")
            if name != boundary.name
                continue
            end
            boundaryTemperatureFields[index].values = zeros(boundary.nFaces)
            boundaryTemperatureFields[index].nFaces = boundary.nFaces
            pointer += 2
            while !occursin("}", lines[pointer])
                vals = match(r"(\w+)\s+(\w+)\s?(\d+)?;", lines[pointer])
                if vals[1] == "type"
                    boundaryTemperatureFields[index].type = vals[2]
                    println("iBoundary: $index, type: $(vals[2])")
                    if vals[2] == "empty"
                        boundaryTemperatureFields[index].values = []
                        boundaryTemperatureFields[index].nFaces = 0
                    end
                elseif vals[1] == "value" && vals[2] == "uniform"
                    temp = tryparse(Float64, vals[3])
                    boundaryTemperatureFields[index].values = [temp for x in boundaryTemperatureFields[index].values]
                end
                pointer += 1
            end
            pointer += 1
        end
    end
    return boundaryTemperatureFields
end # function readTemperatureField


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
    line = readlines(neighborsFileName)[20]
    numNeighbors = tryparse(Int, "$(line[1])")
    s = split((line[3:(end-1)]), " ")
    nbs = map(s -> tryparse(Int, s), s)
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
        for iFace in iFaces[iNeighbors]
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


import LinearSolve as LS
function heatInduction()
    caseDirectory = "/Users/peter/clones/FVM-CFD-prototype/cases/heat-conduction/2D-heat-conduction-on-a-2-by-2-mesh"
    mesh = readOpenFoamMesh(caseDirectory)
    # Define the thermal conductivity and source term
    thermalConductivity = ones(size(mesh.faces)[1])
    numCells = size(mesh.cells)[1]
    heatSource = zeros(numCells)
    # Read initial condition and boundary conditions
    internalTemperatureField = Field(numCells, zeros(numCells))
    boundaryTemperatureFields = readTemperatureField(caseDirectory, mesh, internalTemperatureField)
    # Assemble the coefficient matrix and RHS vector
    matrix, RHS = cellBasedAssembly(mesh, heatSource, thermalConductivity, boundaryTemperatureFields)
    display(matrix)
    display(RHS)
    prob = LS.LinearProblem(matrix, RHS)
    sol = LS.solve(prob)
    display(sol)
end # function main

function cellBasedAssembly(mesh::Mesh, source::Vector{Float64}, diffusionCoeff::Vector{Float64}, boundaryFields::Vector{BoundaryField})::Tuple{Matrix{Float64},Vector{Float64}}
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    coeffMatrix = Matrix(zeros(nCells, nCells))
    for iElement in 1:nCells
        theElement = mesh.cells[iElement]
        RHS[iElement] = source[iElement] * theElement.volume
        diag = 0.0
        nFaces = size(theElement.iFaces)[1]
        for iFace in 1:nFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                fluxCn = 0.0
                fluxFn = 0.0
                fluxCn = diffusionCoeff[iFaceIndex] * theFace.gDiff
                fluxFn = -fluxCn
                coeffMatrix[iElement, theElement.iNeighbors[iFace]] = fluxFn
                diag += fluxCn
            else
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                boundaryType = boundaryFields[iBoundary].type
                fluxCn = 0.0
                fluxFn = 0.0
                if boundaryType == "fixedValue"
                    fluxCb = diffusionCoeff[iFaceIndex] * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb = -fluxCb * boundaryFields[iBoundary].values[relativeFaceIndex]
                    diag += fluxCb
                    RHS[iElement] -= fluxVb
                elseif boundaryType == "zeroGradient"
                end
            end
        end
        coeffMatrix[iElement, iElement] = diag
    end
    return coeffMatrix, RHS
end # function cellBasedAssembly


heatInduction()
end # module HeatInduction