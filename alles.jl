import Pkg
Pkg.activate(".")
using LinearAlgebra
using BenchmarkTools
using StaticArrays
using ElasticArrays
using .Threads
using SparseArrays
using ProgressBars

struct CaseDirError <: Exception
    message::String
end # struct CaseDirError

mutable struct Face
    index::Int
    iNodes::Vector{Int}
    iOwner::Int
    iNeighbor::Int
    centroid::MVector{3,Float32}
    Sf::MVector{3,Float32}
    area::Float32
    gDiff::Float32
    patchIndex::Int
end # struct Face

struct Boundary
    name::String
    type::String
    nFaces::Int
    startFace::Int
end # struct Boundary

mutable struct Cell
    index::Int
    numIntFaces::Int
    volume::Float32
    iFaces::Vector{Int}
    iNeighbors::Vector{Int}
    faceSigns::Vector{Int}
    centroid::MVector{3,Float32}
    # iNodes::Vector{Int}
    # numNeighbors::Int
    # oldVolume::Float32
end # struct Cell

struct Mesh
    nodes::MVector
    faces::Vector{Face}
    boundaries::Vector{Boundary}
    numCells::Int
    cells::Vector{Cell}
    numInteriorFaces::Int
    numBoundaryCells::Int
    numBoundaryFaces::Int
end # struct Mesh

mutable struct Field
    values::Vector{MVector{3,Float32}}
end # struct Field

mutable struct BoundaryField{T}
    name::String
    nFaces::Int
    values::Vector{T}
    type::String
end # struct BoundaryField

struct MatrixAssemblyInput{T}
    mesh::Mesh
    source::Vector{Float32}
    diffusionCoeff::Vector{Float32}
    boundaryFields::Vector{BoundaryField{T}}
end

struct LdcMatrixAssemblyInput
    mesh::Mesh
    nu::Float32
    U::Tuple{Vector{BoundaryField},Field}
end

struct GenericMatrixAssemblyInput
    mesh::Mesh
    sources::Vector{Vector{Float32}}
    variables::Vector{Float32}
    boundaryFields::Vector{Vector{BoundaryField}}
    mappings::Dict{Int,String}
end


function readOpenFoamMesh(caseDir::String)::Mesh
    if !isdir(caseDir)
        throw(CaseDirError("Case Directory '$(caseDir)' does not exist"))
    end
    polymeshDir = joinpath(caseDir, "constant/polyMesh")
    if !isdir(polymeshDir)
        throw(CaseDirError("PolyMesh Directory '$(caseDir)' does not exist"))
    end
    # println("Reading Points..")
    nodes = readPointsFile(polymeshDir)
    # println("Reading Owners..")
    owner = readOwnersFile(polymeshDir)
    # println("Reading Faces..")
    faces = readFacesFile(polymeshDir, owner)
    # println("Reading Neighbors..")
    numNeighbors, faces = readNeighborsFile(polymeshDir, faces)
    numCells = maximum(owner)
    # println("Reading Boundaries..")
    boundaries = readBoundaryFile(polymeshDir)
    # println("Constructing Cells..")
    mesh = constructCells(nodes, boundaries, faces, numCells, numNeighbors)
    # mesh = setupNodeConnectivities(mesh)
    mesh = processOpenFoamMesh(mesh)
    return mesh
end # function readOpenFoamMesh

function getAmountAndStart(file::String)::Tuple{Int,Int}
    for (j, line) in enumerate(eachline(file))
        amount = tryparse(Int, line)
        if isnothing(amount)
            continue
        end
        return amount, j + 2
    end
end

function readPointsFile(polyMeshDir::String)
    pointsFileName = joinpath(polyMeshDir, "points")
    if !isfile(pointsFileName)
        throw(CaseDirError("Points file '$(pointsFileName)' does not exist."))
    end
    nNodes, start = getAmountAndStart(pointsFileName)
    centroids = MVector{nNodes,SVector{3,Float32}}(undef)
    i = 1
    for (j, line) in enumerate(eachline(pointsFileName))
        if j < start
            continue
        end
        if line == ")"
            break
        end
        s = split((line[2:(end-1)]), " ")
        centroids[i] = SVector{3,Float32}(tryparse.(Float32, s))
        i += 1
    end
    return centroids
end # function readPointsFile

function readOwnersFile(polyMeshDir::String)
    ownersFileName = joinpath(polyMeshDir, "owner")
    if !isfile(ownersFileName)
        throw(CaseDirError("Owners file '$(ownersFileName)' does not exist."))
    end
    nOwners, start = getAmountAndStart(ownersFileName)
    owners::Vector{Int} = zeros(nOwners)
    i = 1
    for (j, line) in enumerate(eachline(ownersFileName))
        if j < start
            continue
        end
        if line == ")"
            break
        end
        owners[i] = parse(Int, line) + 1
        i += 1
    end
    return owners
end # function readPointsFile

function readFacesFile(polyMeshDir::String, owners::Vector{Int})
    facesFileName = joinpath(polyMeshDir, "faces")
    if !isfile(facesFileName)
        throw(CaseDirError("Faces file '$(facesFileName)' does not exist."))
    end
    nFaces, start = getAmountAndStart(facesFileName)
    faces = Vector{Face}(undef, nFaces)
    i = 1
    for (j, line) in enumerate(eachline(facesFileName))
        if j < start
            continue
        end
        if line == ")"
            break
        end
        faces[i] = Face(
            i,
            parse.(Int, split(line[3:(end-1)], " ")) .+ 1,
            owners[i],
            -1,
            MVector{3,Float32}(0.0, 0.0, 0.0),
            MVector{3,Float32}(0.0, 0.0, 0.0),
            0.0,
            0.0,
            -1,
        )
        i += 1
    end
    return faces
end # function readPointsFile


function readNeighborsFile(polyMeshDir::String, faces::Vector{Face})::Tuple{Int,Vector{Face}}
    neighborsFileName = joinpath(polyMeshDir, "neighbour")
    if !isfile(neighborsFileName)
        throw(CaseDirError("Neighbors file '$(neighborsFileName)' does not exist."))
    end
    numNeighbors, start = getAmountAndStart(neighborsFileName)
    i = 1
    for (j, line) in enumerate(eachline(neighborsFileName))
        if j < start
            continue
        end
        if line == ")"
            break
        end
        faces[i].iNeighbor = parse(Int, line) + 1
        i += 1
    end
    # faces[j-start+1].iNeighbor = parse(Int, line)
    return numNeighbors, faces
end # function readPointsFile

function readBoundaryFile(polyMeshDir::String)::Vector{Boundary}
    boundariesFileName = joinpath(polyMeshDir, "boundary")
    if !isfile(boundariesFileName)
        throw(CaseDirError("Boundary file '$(boundariesFileName)' does not exist."))
    end
    numBoundaries, start = getAmountAndStart(boundariesFileName)
    i = 17

    lines = readlines(boundariesFileName)
    while isempty(lines[i]) || startswith(lines[i], "//")
        i += 1
    end
    boundaries::Vector{Boundary} = []
    lines = readlines(boundariesFileName)[i+2:(end-4)]

    l = split(join(lines), "}")
    for iBound in 1:numBoundaries
        name = ""
        type = ""
        nFaces = ""
        startface = ""
        spl = split(l[iBound], "{")
        name = strip(spl[1])
        arrayofpairs = split(spl[2], ';')
        for pair in arrayofpairs
            if strip(pair) == ""
                continue
            end
            pair = lstrip(pair)
            p = split(replace(pair, r"\s+" => "#"), "#")
            left = p[1]
            right = p[2]
            if left == "type"
                type = right
            elseif left == "inGroups"
            elseif left == "nFaces"
                nFaces = tryparse(Int, right)
            elseif left == "startFace"
                startface = tryparse(Int, right)
            else
                println("unknown boundary key $left")
            end
        end
        boundary = Boundary(
            name,
            type,
            nFaces,
            startface
        )

        push!(boundaries, boundary)
    end
    return boundaries
end # function readPointsFile

function constructCells(nodes::MVector, boundaries::Vector{Boundary}, faces::Vector{Face}, numCells::Int, numInteriorFaces::Int)::Mesh
    cells = [
        Cell(
            index,
            0,
            0.0,
            [],
            [],
            [],
            MVector{3,Float32}(0.0, 0.0, 0.0),
        )
        for index in 1:numCells
    ]
    numInteriors = numInteriorFaces
    for interiorFace::Int in 1:numInteriorFaces
        iOwner = faces[interiorFace].iOwner
        iNeighbor = faces[interiorFace].iNeighbor
        if iNeighbor <= 0
            # println("settings numinteriors to $numInteriors")
            numInteriors = interiorFace - 1
            break
        end
        push!(cells[iOwner].iFaces, faces[interiorFace].index)
        push!(cells[iOwner].iNeighbors, iNeighbor)
        push!(cells[iOwner].faceSigns, 1)
        push!(cells[iNeighbor].iFaces, faces[interiorFace].index)
        push!(cells[iNeighbor].iNeighbors, iOwner)
        push!(cells[iNeighbor].faceSigns, -1)
        cells[iOwner].numIntFaces += 1
        cells[iNeighbor].numIntFaces += 1
    end
    for boundaryFace in (numInteriors+1):size(faces)[1]
        owner = faces[boundaryFace].iOwner
        push!(cells[owner].iFaces, boundaryFace)
        push!(cells[owner].faceSigns, 1)
    end

    # for cell in cells
    #     cell.numNeighbors = size(cell.iNeighbors)[1]
    # end
    numBoundaryCells = size(faces)[1] - numInteriors
    numBoundaryFaces = size(faces)[1] - numInteriors
    return Mesh(nodes, faces, boundaries, numCells, cells, numInteriors, numBoundaryCells, numBoundaryFaces)
end # function constructCells

function setupNodeConnectivities(mesh::Mesh)::Mesh
    for iFace in 1:size(mesh.faces)[1]
        iNodes = mesh.faces[iFace].iNodes
        nNodes = size(iNodes)[1]
        for iNode in 1:nNodes
            push!(mesh.nodes[iNodes[iNode]].iFaces, mesh.faces[iFace].index)
        end
    end
    # for cell in mesh.cells
    #     for face in mesh.faces[cell.iFaces]
    #         for iNode in face.iNodes
    #             if !(iNode in cell.iNodes)
    #                 push!(cell.iNodes, iNode)
    #                 push!(mesh.nodes[iNode].iCells, cell.index)
    #             end
    #         end
    #     end
    # end
    return mesh
end



function processOpenFoamMesh(mesh::Mesh)::Mesh
    mesh = processBasicFaceGeometry(mesh)
    mesh = computeElementVolumeAndCentroid(mesh)
    mesh = processSecondaryFaceGeometry(mesh)
    # mesh = sortBoundaryNodesFromInteriorNodes(mesh)
    mesh = labelBoundaryFaces(mesh)
    return mesh
end # function processOpenFoamMesh

function magnitude(vector::MVector{3,Float32})::Float32
    return sqrt(vector'vector)
end # function magnitude

function processBasicFaceGeometry(mesh::Mesh)::Mesh
    for face in mesh.faces
        area = 0.0
        # special case: triangle
        if size(face.iNodes)[1] == 3
            # sum x,y,z and divide by 3
            triangleNodes = [mesh.nodes[i] for i in face.iNodes]
            face.centroid .= sum(triangleNodes) / 3
            face.Sf .= 0.5 * cross(triangleNodes[2] - triangleNodes[1], triangleNodes[3] - triangleNodes[1])
            area = magnitude(face.Sf)
        else # general case, polygon is not a triangle 
            nodes = [mesh.nodes[i] for i in face.iNodes]
            center = sum(nodes) / size(nodes)[1]
            # Using the center to compute the area and centroid of virtual
            # triangles based on the center and the face nodes
            triangleNode1 = center
            triangleNode3 = MVector{3,Float32}(0.0, 0.0, 0.0)
            for (iNodeIndex, iNode) in enumerate(face.iNodes)
                if iNodeIndex < size(face.iNodes)[1]
                    triangleNode3 .= mesh.nodes[face.iNodes[iNodeIndex+1]]
                else
                    triangleNode3 .= mesh.nodes[face.iNodes[1]]
                end
                # Calculate the centroid of a given subtriangle
                localCentroid = (triangleNode1 .+ mesh.nodes[iNode] .+ triangleNode3) ./ 3
                # Calculate the surface area vector of a given subtriangle by cross product
                localSf = 0.5 .* cross(mesh.nodes[iNode] .- triangleNode1, triangleNode3 .- triangleNode1)
                # Calculate the surface area of a given subtriangle
                localArea = sqrt(dot(localSf, localSf))
                face.centroid .+= localArea * localCentroid
                face.Sf += localSf
            end
            area = magnitude(face.Sf)
            # Compute centroid of the polygon
            face.centroid ./= area
        end
        face.area = area
    end
    return mesh
end # function processBasicFaceGeometry

function computeElementVolumeAndCentroid(mesh)::Mesh
    for iElement in 1:(size(mesh.cells)[1])
        iFaces = mesh.cells[iElement].iFaces
        elementCenter = MVector{3,Float32}(0.0, 0.0, 0.0)
        for iFace in iFaces
            elementCenter .+= mesh.faces[iFace].centroid
        end
        elementCenter ./= size(iFaces)[1]
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
        # mesh.cells[iElement].oldVolume = localVolumeSum
    end
    return mesh
end # function computeElementVolumeAndCentroid

function processSecondaryFaceGeometry(mesh::Mesh)::Mesh
    # Loop over interior faces
    eCN = MVector{3,Float32}(0.0, 0.0, 0.0)
    CN = MVector{3,Float32}(0.0, 0.0, 0.0)

    for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        # Compute unit surtheFace normal vector
        # nf = theFace.Sf ./ theFace.area
        ownerElement = mesh.cells[theFace.iOwner]
        neighborElement = mesh.cells[theFace.iNeighbor]

        # vector between cell centroids
        CN = neighborElement.centroid .- ownerElement.centroid
        # length of said vector / direct distance between cell centroids
        magCN = magnitude(CN)
        # normalized vector to neighbor
        eCN .= CN ./ magCN

        E = theFace.area * eCN
        # essentially strength of flux in the direction of neighboring cell
        # eq 8.11: gdiff of face = ||face Surface || / || distance between cell centroids ||
        theFace.gDiff = magnitude(E) / magCN
        # theFace.T = theFace.Sf - E

        # not needed?
        # Compute theFace weighting factor
        # Cf = theFace.centroid .- ownerElement.centroid
        # fF = neighborElement.centroid - theFace.centroid
        # theFace.gf = dot(Cf, nf) / (dot(Cf, nf) + dot(fF, nf))
    end
    for iBFace in (mesh.numInteriorFaces+1):size(mesh.faces)[1]
        theBFace = mesh.faces[iBFace]
        ownerElement = mesh.cells[theBFace.iOwner]
        CN .= theBFace.centroid .- ownerElement.centroid
        # mesh.faces[iBFace].CN = CN
        mesh.faces[iBFace].gDiff = theBFace.area * theBFace.area / dot(CN, theBFace.Sf)
        # magCN = magnitude(CN)
        # mesh.faces[iBFace].magCN = magCN
        # eCN = (1 / magCN) * CN
        # E = theBFace.area * eCN
        # mesh.faces[iBFace].T = theBFace.Sf - E
        # mesh.faces[iBFace].gf = 1.0
        # mesh.faces[iBFace].walldist = dot(CN, theBFace.Sf) / magnitude(theBFace.Sf)
    end
    # for iElement in mesh.cells
    #     iFaces = iElement.iFaces
    #     iNeighbors = iElement.iNeighbors
    #     kf = 1
    #     for i in 1:size(iNeighbors)[1]
    #         iFace = iFaces[i]
    #         if mesh.faces[iFace].iOwner == iElement
    #             mesh.faces[iFace].iOwnerNeighborCoef = kf
    #         elseif mesh.faces[iFace].iNeighbor == iElement
    #             mesh.faces[iFace].iNeighborOwnerCoef = kf
    #         end
    #         kf += 1
    #     end
    # end
    return mesh
end # function processSecondaryFaceGeometry

function sortBoundaryNodesFromInteriorNodes(mesh::Mesh)::Mesh
    for face in mesh.faces[1:mesh.numInteriorFaces]
        for iNode in face.iNodes
            mesh.nodes[iNode:iNode+2].flag = 1
        end
    end
    for boundary in mesh.boundaries
        startFace = boundary.startFace
        nBFaces = boundary.nFaces
        s1 = boundary.name == "frontAndBack"
        s2 = boundary.name == "frontAndBackPlanes"
        for face in mesh.faces[startFace:(startFace+nBFaces)]
            for iNode in face.iNodes
                mesh.nodes[iNode:iNode+2].flag = (s1 || s2) ? 1 : 0
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

function readFields(dir::String, mesh::Mesh)::Tuple{Vector{Vector{BoundaryField}},Dict{Int,String}}
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
function parseVec(string::String)::Vector{Float32}
    cp = replace(string, "(" => "")
    cp = replace(cp, ")" => "")
    cp = replace(cp, ";" => "")
    cp = strip(cp)
    return parse.(Float32, split(cp, " "))
end
parseVec(sub::SubString)::Vector{Float32} = parseVec(String(sub))

function isUniformInternalField(file::String)::Bool
    for line in eachline(file)
        if startswith(line, "internalField")
            return String(match(r"^(\w+)\s+(\w+)", line)[2]) == "uniform"
        end
    end
end

function getFieldType(file::String)
    for line in eachline(file)
        if contains(line, "class")
            if contains(line, "volVectorField")
                return MVector{3,Float32}
            end
            return Float32
        end
    end
end
"""
returns either Float32 or MVector{3, Float32}
"""
function getUniformValue(file::String)
    for line in eachline(file)
        if startswith(line, "internalField")
            cpy = replace(line, "internalField   uniform " => "")
            cpy = replace(cpy, "(" => "")
            cpy = replace(cpy, ")" => "")
            cpy = replace(cpy, ";" => "")
            values = parse.(Float32, split(cpy, " "))
            if size(values)[1] == 1
                return values[1]
            end
            return MVector(values[1], values[2], values[3])
        end
    end
end

function isvec3(file::String)::Bool
    for line in eachline(file)
        if j < start
            continue
        end
        return startswith(line, "(")
    end
end

function readField(filePath::String, mesh::Mesh)
    fileName = rsplit(filePath, "/", limit=1)[1]
    if !isfile(filePath)
        throw(CaseDirError("Field file '$(filePath)' does not exist."))
    end
    isUniform = isUniformInternalField(filePath)
    # TODO so far not used 
    numCells = size(mesh.cells)[1]
    joined = ""
    fieldType = MVector{3,Float32} # getFieldType(filePath)
    internalValues = Vector{fieldType}(undef, numCells)
    if isUniform
        # println("isuniform")
        value = getUniformValue(filePath)
        internalValues = fill(value, numCells)
        lines = readlines(filePath)
        i = 16
        while !contains(lines[i], "boundaryField")
            i += 1
        end
        boundaryLines = lines[i+2:end-4]
        joined = join(boundaryLines)
    else
        doBoundaries = false
        _, start = getAmountAndStart(filePath)
        index = 1
        if fieldType == MVector{3,Float32}
            for (j, line) in enumerate(eachline(filePath))
                if j < start
                    continue
                end
                if line == ")"
                    doBoundaries = true
                end
                if !doBoundaries
                    internalValues[index] = parse.(Float32, split(line[2:end-1], " "))
                    index += 1
                else
                    joined = string(joined, line)
                end
            end
        else
            for (j, line) in enumerate(eachline(filePath))
                if j < start
                    continue
                end
                if line == ")"
                    doBoundaries = true
                end
                if !doBoundaries
                    internalValues[index] = parse(Float32, line)
                else
                    joined = string(joined, line)
                end
            end
        end
    end
    internalField = Field(internalValues)
    splitted = split(joined, "}")
    boundaryFields = []
    for boundary in mesh.boundaries
        for b in splitted
            matches = match(r"\s*(\w+)\s*\{\s*type\s*(\w+);(?:\s*value\s*(\w+) (?:(\d+)|(\([\d\s]+\)));)?", b)
            if isnothing(matches)
                continue
            end
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
                    scalar = tryparse(Float32, matches[4])
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
    return boundaryFields, internalField
end

function readPropertiesFile(path::String)::Float32
    # println("Reading $(path)")
    if !isfile(path)
        throw(CaseDirError("Field file '$(path)' does not exist."))
    end
    file = read(path, String)
    variable = match(r"\w+\s*\[[\s\d-]+\]\s*([\.\d\-e]+);", file)
    val = tryparse(Float32, variable[1])
    return val
end


function heatInduction(caseDirectory::String)::MatrixAssemblyInput
    mesh = readOpenFoamMesh(caseDirectory)
    # Define the thermal conductivity and source term
    thermalConductivity = ones(size(mesh.faces)[1])
    numCells = size(mesh.cells)[1]
    heatSource = zeros(numCells)

    # Read initial condition and boundary conditions
    boundaryTemperatureFields = readField(joinpath(caseDirectory, "0/T"), mesh)

    # Assemble the coefficient matrix and RHS vector
    matrixAssemblyInput = MatrixAssemblyInput(mesh, heatSource, thermalConductivity, boundaryTemperatureFields)
    return matrixAssemblyInput
end # function heatInduction

function genericHeatInduction(caseDirectory::String)::GenericMatrixAssemblyInput
    mesh = readOpenFoamMesh(caseDirectory)
    # Define the thermal conductivity and source term
    thermalConductivity = ones(size(mesh.faces)[1])
    numCells = size(mesh.cells)[1]
    heatSource = zeros(numCells)

    # Read initial condition and boundary conditions
    boundaryTemperatureFields = readField(joinpath(caseDirectory, "0/T"), mesh)
    mapping = Dict()
    mapping[1] = "T"
    # Assemble the coefficient matrix and RHS vector
    matrixAssemblyInput = GenericMatrixAssemblyInput(mesh, [heatSource], [thermalConductivity], boundaryTemperatureFields, mapping)
    return matrixAssemblyInput
end # function heatInduction

function LidDrivenCavity(caseDirectory::String)::LdcMatrixAssemblyInput
    mesh = readOpenFoamMesh(caseDirectory)
    # Define the thermal conductivity and source term
    nu = readPropertiesFile(joinpath(caseDirectory, "constant/transportProperties"))
    # Read initial condition and boundary conditions
    # p = readField(joinpath(caseDirectory, "0/p"), mesh)
    U = readField(joinpath(caseDirectory, "0/U"), mesh)
    # Assemble the coefficient matrix and RHS vector
    ldcInput = LdcMatrixAssemblyInput(mesh, nu, U)
    return ldcInput
end # function LidDrivenCavity


function ThreadedCellBasedAssembly(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nCells = size(mesh.cells)[1]
    RHS = zeros(Float32, nCells * 3)
    entriesNeeded, offsets = estimate_data(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    vals = Vector{Float32}(undef, entriesNeeded * 3)
    chunks = Iterators.partition(1:nCells, nCells ÷ nthreads())
    chunks = [c for c in chunks]
    @threads for chunk in chunks
        CellBasedHelper(chunk, input, RHS, rows, cols, offsets, vals)
    end
    return rows, cols, vals
end

function CellBasedHelper(chunk::UnitRange, input::LdcMatrixAssemblyInput, RHS::Vector{Float32}, rows::Vector{Int32}, cols::Vector{Int32}, offsets, vals::Vector)
    mesh = input.mesh
    nu = input.nu
    nCells = size(mesh.cells)[1]
    velocity_boundary = input.U[1]
    velocity_internal = input.U[2].values
    for iElement in chunk
        theElement = mesh.cells[iElement]
        numFaces = size(theElement.iFaces)[1]
        @inbounds diagx::Float32 = velocity_internal[iElement][1]
        @inbounds diagy::Float32 = velocity_internal[iElement][2]
        @inbounds diagz::Float32 = velocity_internal[iElement][3]
        @inbounds for iFace in 1:theElement.numIntFaces
            @inbounds iFaceIndex = theElement.iFaces[iFace]
            @inbounds theFace = mesh.faces[iFaceIndex]
            fluxCn = nu * theFace.gDiff
            fluxFn = -fluxCn
            idx = offsets[iElement] + iFace
            @inbounds cols[idx] = iElement
            @inbounds rows[idx] = theElement.iNeighbors[iFace]
            @inbounds vals[idx] = fluxFn    # x  
            @inbounds vals[2*idx] = fluxFn  # y
            @inbounds vals[3*idx] = fluxFn  # z
            diagx += fluxCn
            diagy += fluxCn
            diagz += fluxCn
        end
        for iFace in theElement.numIntFaces+1:nFaces
            @inbounds iFaceIndex = theElement.iFaces[iFace]
            @inbounds theFace = mesh.faces[iFaceIndex]
            @inbounds iBoundary = mesh.faces[iFaceIndex].patchIndex
            @inbounds boundaryType = velocity_boundary[iBoundary].type
            if boundaryType == "fixedValue"
                fluxCb = nu * theFace.gDiff
                relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                fluxVb::Vector{Float32} = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCb
                @inbounds RHS[iElement] -= fluxVb[1]
                @inbounds RHS[iElement+nCells] -= fluxVb[2]
                @inbounds RHS[iElement+nCells+nCells] -= fluxVb[3]
                diagx += fluxCb
                diagy += fluxCb
                diagz += fluxCb
            end
        end
        idx = offsets[iElement]
        @inbounds cols[idx] = iElement
        @inbounds rows[idx] = iElement
        @inbounds vals[idx] = diagx    # x  
        @inbounds vals[2*idx] = diagy  # y
        @inbounds vals[3*idx] = diagz  # z
    end
end


function CellBasedAssembly(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    velocity_internal = input.U[2].values
    nCells = size(mesh.cells)[1]
    RHS = zeros(Float32, nCells * 3)
    entriesNeeded, offsets = estimate_data(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    vals = Vector{Float32}(undef, entriesNeeded * 3)
    idx = 1
    cellIdx = 1
    for iElement = 1:nCells
        set = false
        theElement = mesh.cells[iElement]
        numFaces = size(theElement.iFaces)[1]
        @inbounds diagx::Float32 = velocity_internal[iElement][1]
        @inbounds diagy::Float32 = velocity_internal[iElement][2]
        @inbounds diagz::Float32 = velocity_internal[iElement][3]
        for iFace in 1:theElement.numIntFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            fluxCn = nu * theFace.gDiff
            fluxFn = -fluxCn
            if theElement.iNeighbors[iFace] > iElement && !set
                cellIdx = idx
                idx += 1
                set = true
            end
            cols[idx] = iElement
            rows[idx] = theElement.iNeighbors[iFace]
            vals[idx] = fluxFn    # x  
            vals[idx+entriesNeeded] = fluxFn  # y
            vals[idx+entriesNeeded+entriesNeeded] = fluxFn  # z
            idx += 1
            diagx += fluxCn
            diagy += fluxCn
            diagz += fluxCn
        end
        if !set
            cellIdx = idx
        end
        for iFace in theElement.numIntFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            fluxCb = nu * theFace.gDiff
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            fluxVb::Vector{Float32} = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCb
            RHS[iElement] -= fluxVb[1]
            RHS[iElement+nCells] -= fluxVb[2]
            RHS[iElement+nCells+nCells] -= fluxVb[3]
            diagx += fluxCb
            diagy += fluxCb
            diagz += fluxCb
        end
        cols[cellIdx] = iElement
        rows[cellIdx] = iElement
        vals[cellIdx] = diagx    # x  
        vals[cellIdx+entriesNeeded] = diagy  # y
        vals[cellIdx+entriesNeeded+entriesNeeded] = diagz  # z
    end
    return rows, cols, vals, RHS
end # function cellBasedAssemblySparseMultiVectorPrealloc

function estimate_data(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    e2 = size(mesh.cells)[1] + 2 * mesh.numInteriorFaces
    nCells = size(mesh.cells)[1]

    offsets::Vector{Int32} = ones(e2 + 1)
    offsets[1] = 1
    for iElement in 2:nCells
        offsets[iElement] += offsets[iElement-1] + mesh.cells[iElement-1].numIntFaces
    end
    offsets[end] = e2 + 1
    return e2, offsets
end

function estimate_data_facebased(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    velocity_internal = input.U[2].values
    e2 = size(mesh.cells)[1] + 2 * mesh.numInteriorFaces
    nCells = size(mesh.cells)[1]
    vals = Vector{Float32}(undef, e2 * 3)

    offsets::Vector{Int32} = ones(e2 + 1)
    offsets[1] = 1
    vals[1] = velocity_internal[1][1]
    vals[1+e2] = velocity_internal[1][2]
    vals[1+e2+e2] = velocity_internal[1][3]
    ## TODO fix offsets, does not make sense for face based assembly        
    for iElement in 2:nCells
        offsets[iElement] += offsets[iElement-1] + mesh.cells[iElement-1].numIntFaces
        vals[iElement] = velocity_internal[iElement][1]
        vals[iElement+e2] = velocity_internal[iElement][2]
        vals[iElement+e2+e2] = velocity_internal[iElement][3]
    end
    offsets[end] = e2 + 1
    return e2, offsets, vals
end

function bench_gc(input::LdcMatrixAssemblyInput, func::Function, case::String, runGC::Bool)
    times = []
    for _ in 1:10
        if runGC
            GC.gc()
        end
        start = time()
        func(input)
        dur = time() - start
        push!(times, dur)
    end
    long = if case == "LDC-S"
        "Lid-Driven Cavity S"
    elseif case == "LDC-M"
        "Lid-Driven Cavity M"
    else
        "WindsorBody"
    end
    ms = mean(times) * 1000
    med = median(times) * 1000
    included = if runGC
        "false"
    else
        "true"
    end
    println("$ms,$case,$long,$(Threads.nthreads()),$(String(Symbol(func))),$included,mean,julia")
    println("$med,$case,$long,$(Threads.nthreads()),$(String(Symbol(func))),$included,median,julia")
end



function FaceBasedAssembly(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    nCells = size(mesh.cells)[1]
    RHS = zeros(Float32, nCells * 3)
    entriesNeeded = size(mesh.cells)[1] + 2 * mesh.numInteriorFaces

    # entriesNeeded, offsets, vals = estimate_data_facebased(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)

    @inbounds for (iFace, theFace) in enumerate(mesh.faces)
        fluxCn = nu * theFace.gDiff
        if theFace.iNeighbor > 0
            iOwner = theFace.iOwner
            iNeighbor = theFace.iNeighbor
            fluxFn = -fluxCn
            idx = offsets[iOwner] + seenOwner[iOwner]
            seenOwner[iOwner] += 1
            setIndex!(iOwner, iNeighbor, fluxFn, fluxFn, fluxFn, rows, cols, vals, idx, entriesNeeded)
            idx = offsets[iNeighbor] + seenOwner[iNeighbor]
            seenOwner[iNeighbor] += 1
            setIndex!(iNeighbor, iOwner, fluxFn, fluxFn, fluxFn, rows, cols, vals, idx, entriesNeeded)
            idx = offsets[iOwner]
            setIndex!(iOwner, iOwner, fluxCn, fluxCn, fluxCn, rows, cols, vals, idx, entriesNeeded)
            idx = offsets[iNeighbor]
            setIndex!(iNeighbor, iNeighbor, fluxCn, fluxCn, fluxCn, rows, cols, vals, idx, entriesNeeded)
        else
            @inbounds iBoundary = mesh.faces[iFace].patchIndex
            @inbounds boundaryType = velocity_boundary[iBoundary].type
            if boundaryType == "fixedValue"
                relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
                fluxVb::Vector{Float32} = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCn
                idx = offsets[theFace.iOwner]
                setIndex!(theFace.iOwner, theFace.iOwner, fluxCn, fluxCn, fluxCn, rows, cols, vals, idx, entriesNeeded)
                @inbounds RHS[theFace.iOwner] -= fluxVb[1]
                @inbounds RHS[theFace.iOwner+nCells] -= fluxVb[2]
                @inbounds RHS[theFace.iOwner+nCells+nCells] -= fluxVb[3]
            end
        end
    end
    return rows, cols, vals
end # function faceBasedAssembly

function setIndex!(ir, ic, valx, valy, valz, rows, cols, vals, idx, entries)
    # coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxC
    @inbounds cols[idx] = ic
    @inbounds rows[idx] = ir
    @inbounds vals[idx] += valx    # x  
    @inbounds vals[idx+entries] += valy  # y
    @inbounds vals[idx+entries+entries] += valz  # z
end

function BatchedFaceBasedAssembly(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]

    nCells = size(mesh.cells)[1]
    RHS = zeros(Float32, nCells * 3)
    entriesNeeded, offsets, vals = estimate_data_facebased(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        fluxCn = nu * theFace.gDiff
        fluxFn = -fluxCn
        idx = offsets[iOwner] + seenOwner[iOwner]
        seenOwner[iOwner] += 1
        setIndex!(iOwner, iNeighbor, fluxFn, fluxFn, fluxFn, rows, cols, vals, idx, entriesNeeded)
        idx = offsets[iNeighbor] + seenOwner[iNeighbor]
        seenOwner[iNeighbor] += 1
        setIndex!(iNeighbor, iOwner, fluxFn, fluxFn, fluxFn, rows, cols, vals, idx, entriesNeeded)
        idx = offsets[iOwner]
        setIndex!(iOwner, iOwner, fluxCn, fluxCn, fluxCn, rows, cols, vals, idx, entriesNeeded)
        idx = offsets[iNeighbor]
        setIndex!(iNeighbor, iNeighbor, fluxCn, fluxCn, fluxCn, rows, cols, vals, idx, entriesNeeded)
    end

    @inbounds for iBoundary in eachindex(mesh.boundaries)
        if velocity_boundary[iBoundary].type != "fixedValue"
            continue
        end
        @inbounds theBoundary = mesh.boundaries[iBoundary]
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        @inbounds for iFace in startFace:endFace-1
            @inbounds theFace = mesh.faces[iFace]
            fluxCn = nu * theFace.gDiff
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            @inbounds fluxVb = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCn
            @inbounds idx = offsets[theFace.iOwner]
            setIndex!(theFace.iOwner, theFace.iOwner, fluxCn, fluxCn, fluxCn, rows, cols, vals, idx, entriesNeeded)
            @inbounds RHS[theFace.iOwner] -= fluxVb[1]
            @inbounds RHS[theFace.iOwner+nCells] -= fluxVb[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= fluxVb[3]
        end
    end
    return rows, cols, vals, RHS
end # function batchedFaceBasedAssembly


# if abspath(PROGRAM_FILE) == @__FILE__
#     funcs = [CellBasedAssembly, FaceBasedAssembly, BatchedFaceBasedAssembly]
#     parFuncs = [ThreadedCellBasedAssembly]
#     iterFuncs = if Threads.nthreads() > 1
#         parFuncs
#     else
#         funcs
#     end
#     for c = readdir("cases")
#         case = joinpath("cases", c)
#         input = LidDrivenCavity(case)
#         for assemblyMethod in iterFuncs
#             bench_gc(input, assemblyMethod, c, false)
#             bench_gc(input, assemblyMethod, c, true)
#         end
#     end
# end