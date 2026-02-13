import Pkg
Pkg.activate(".")
using LinearAlgebra
using BenchmarkTools
using StaticArrays
using ElasticArrays
using .Threads
using SparseArrays
using ProgressBars
using CUDA
struct CaseDirError <: Exception
    message::String
end # struct CaseDirError

mutable struct Face
    index::Int32
    iNodes::Vector{Int32}
    iOwner::Int32
    iNeighbor::Int32
    centroid::MVector{3,Float32}
    Sf::MVector{3,Float32}
    area::Float32
    gDiff::Float32
    patchIndex::Int32
    relativeToOwner::Int32
    relativeToNeighbor::Int32
end # struct Face

struct Boundary
    name::String
    type::String
    nFaces::Int32
    startFace::Int32
    index::Int32
end # struct Boundary

mutable struct Cell
    index::Int32
    numIntFaces::Int32
    volume::Float32
    iFaces::Vector{Int32}
    iNeighbors::Vector{Int32}
    faceSigns::Vector{Int32}
    centroid::MVector{3,Float32}
    # iNodes::Vector{Int32}
    # numNeighbors::Int32
    # oldVolume::Float32
end # struct Cell

struct Mesh
    nodes::Vector
    faces::Vector{Face}
    boundaries::Vector{Boundary}
    numCells::Int32
    cells::Vector{Cell}
    numInteriorFaces::Int32
    numBoundaryCells::Int32
    numBoundaryFaces::Int32
end # struct Mesh

mutable struct Field
    values::Vector{MVector{3,Float32}}
end # struct Field

mutable struct BoundaryField
    name::String
    nFaces::Int32
    values::Vector{SVector{3,Float32}}
    type::String
end # struct BoundaryField

struct MatrixAssemblyInput
    mesh::Mesh
    source::Vector{Float32}
    diffusionCoeff::Vector{Float32}
    boundaryFields::Vector{BoundaryField}
end

struct LdcMatrixAssemblyInput
    mesh::Mesh
    nu::Vector{Float32}
    U::Tuple{Vector{BoundaryField},Field}
end

struct GenericMatrixAssemblyInput
    mesh::Mesh
    sources::Vector{Vector{Float32}}
    variables::Vector{Float32}
    boundaryFields::Vector{Vector{BoundaryField}}
    mappings::Dict{Int32,String}
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

function getAmountAndStart(file::String)::Tuple{Int32,Int32}
    for (j, line) in enumerate(eachline(file))
        amount = tryparse(Int32, line)
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
    centroids = Vector{SVector{3,Float32}}(undef, nNodes)
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
    owners::Vector{Int32} = zeros(nOwners)
    i = 1
    for (j, line) in enumerate(eachline(ownersFileName))
        if j < start
            continue
        end
        if line == ")"
            break
        end
        owners[i] = parse(Int32, line) + 1
        i += 1
    end
    return owners
end # function readPointsFile

function readFacesFile(polyMeshDir::String, owners::Vector{Int32})
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
            parse.(Int32, split(line[3:(end-1)], " ")) .+ 1,
            owners[i],
            -1,
            MVector{3,Float32}(0.0, 0.0, 0.0),
            MVector{3,Float32}(0.0, 0.0, 0.0),
            0.0,
            0.0,
            -1,
            0,
            0
        )
        i += 1
    end
    return faces
end # function readPointsFile


function readNeighborsFile(polyMeshDir::String, faces::Vector{Face})::Tuple{Int32,Vector{Face}}
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
        faces[i].iNeighbor = parse(Int32, line) + 1
        i += 1
    end
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
                nFaces = tryparse(Int32, right)
            elseif left == "startFace"
                startface = tryparse(Int32, right)
            else
                println("unknown boundary key $left")
            end
        end
        boundary = Boundary(
            name,
            type,
            nFaces,
            startface,
            iBound
        )

        push!(boundaries, boundary)
    end
    return boundaries
end # function readPointsFile

function constructCells(nodes::Vector, boundaries::Vector{Boundary}, faces::Vector{Face}, numCells::Int32, numInteriorFaces::Int32)::Mesh
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
    for InteriorFace::Int32 in 1:numInteriorFaces
        iOwner = faces[InteriorFace].iOwner
        iNeighbor = faces[InteriorFace].iNeighbor
        if iNeighbor <= 0
            # println("settings numInteriors to $numInteriors")
            numInteriors = InteriorFace - 1
            break
        end
        push!(cells[iOwner].iFaces, faces[InteriorFace].index)
        push!(cells[iOwner].iNeighbors, iNeighbor)
        push!(cells[iOwner].faceSigns, 1)
        push!(cells[iNeighbor].iFaces, faces[InteriorFace].index)
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
    # Loop over Interior faces
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

function readFields(dir::String, mesh::Mesh)::Tuple{Vector{Vector{BoundaryField}},Dict{Int32,String}}
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
function parseVec(string::String)::MVector{3,Float32}
    cp = replace(string, "(" => "")
    cp = replace(cp, ")" => "")
    cp = replace(cp, ";" => "")
    cp = strip(cp)
    return parse.(Float32, split(cp, " "))
end
parseVec(sub::SubString)::Vector{Float32} = parseVec(String(sub))

function isUniforminternalField(file::String)::Bool
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
    isUniform = isUniforminternalField(filePath)
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
            values::Vector{MVector{3,Float32}} = []
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
                    values = [SVector{3,Float32}(vector) for _ in 1:nFaces]
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
    ldcInput = LdcMatrixAssemblyInput(mesh, fill(nu, length(mesh.cells)), U)
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
    entriesNeeded = size(mesh.cells)[1] + 2 * mesh.numInteriorFaces
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
            fluxCn = nu[iElement] * theFace.gDiff
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
            fluxCb = nu[theFace.iOwner] * theFace.gDiff
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
    for iElement in 2:nCells
        offsets[iElement] += offsets[iElement-1] + mesh.cells[iElement-1].numIntFaces
    end
    offsets[end] = e2 + 1
    return e2, offsets
end

function estimate_data_facebased(input::LdcMatrixAssemblyInput)#::Tuple{Vector{Int32},Vector{Float32}}
    mesh = input.mesh
    velocity_internal = input.U[2].values
    entries = size(mesh.cells)[1] + 2 * mesh.numInteriorFaces
    nCells = size(mesh.cells)[1]
    vals = Vector{Float32}(undef, entries * 3)

    offsets::Vector{Int32} = ones(Int32, nCells)
    negOffsets::Vector{Int32} = zeros(Int32, nCells)

    vals[1] = velocity_internal[1][1]
    vals[1+entries] = velocity_internal[1][2]
    vals[1+entries+entries] = velocity_internal[1][3]
    for iElement in 2:nCells
        offsets[iElement] += offsets[iElement-1] + mesh.cells[iElement-1].numIntFaces
        vals[iElement] = velocity_internal[iElement][1]
        vals[iElement+entries] = velocity_internal[iElement][2]
        vals[iElement+entries+entries] = velocity_internal[iElement][3]
    end
    for iElement in 1:nCells
        theElement = mesh.cells[iElement]
        for iFace in 1:theElement.numIntFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor > iElement
                negOffsets[theFace.iNeighbor] += 1
            end
        end
        offsets[iElement] += negOffsets[iElement]  # increase offset
    end
    return offsets, negOffsets, vals
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


function bench_phi(input::LdcMatrixAssemblyInput, case::String, runGC::Bool, phiFunc::Function)
    times = []
    for _ in 1:10
        if runGC
            GC.gc()
        end
        start = time()
        DivLapBatchedFaceBasedAssembly(input, phiFunc)
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
    println("time_ms,case_short,case_long,threads,algorithm,incl_gc,metric,language,interpolationMethod")
    println("$ms,$case,$long,$(Threads.nthreads()),BatchedFaceBasedAssembly,$included,mean,julia,$(String(Symbol(phiFunc)))")
    println("$med,$case,$long,$(Threads.nthreads()),BatchedFaceBasedAssembly,$included,median,julia,$(String(Symbol(phiFunc)))")
end



function FaceBasedAssembly(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    nCells = size(mesh.cells)[1]
    RHS = zeros(Float32, nCells * 3)
    entriesNeeded = size(mesh.cells)[1] + 2 * mesh.numInteriorFaces

    offsets, vals = estimate_data_facebased(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)
    @inbounds for theFace in mesh.faces
        fluxCn = nu * theFace.gDiff
        if theFace.iNeighbor > 0
            iOwner = theFace.iOwner
            iNeighbor = theFace.iNeighbor
            fluxFn = -fluxCn
            # set diag and upper 
            setIndex!(iOwner, iOwner, fluxCn, rows, cols, vals, offsets[iOwner], entriesNeeded)
            setIndex!(iOwner, iNeighbor, fluxFn, rows, cols, vals, offsets[iOwner] + seenOwner[iOwner], entriesNeeded)
            # increment für symmetry 
            seenOwner[iOwner] += 1
            # set diag and lower
            setIndex!(iNeighbor, iNeighbor, fluxCn, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
            setIndex!(iNeighbor, iOwner, fluxFn, rows, cols, vals, offsets[iNeighbor] + seenOwner[iNeighbor], entriesNeeded)
            # increment für symmetry 
            seenOwner[iNeighbor] += 1
        else
            @inbounds iBoundary = theFace.patchIndex
            @inbounds boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = theFace.index - mesh.boundaries[iBoundary].startFace
            fluxVb::Vector{Float32} = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCn
            idx = offsets[theFace.iOwner]
            setIndex!(theFace.iOwner, theFace.iOwner, fluxCn, rows, cols, vals, idx, entriesNeeded)
            @inbounds RHS[theFace.iOwner] -= fluxVb[1]
            @inbounds RHS[theFace.iOwner+nCells] -= fluxVb[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= fluxVb[3]
        end
    end
    return rows, cols, vals
end # function faceBasedAssembly

function setIndex!(ic::Int32, ir::Int32, val::Float32, rows::Vector{Int32}, cols::Vector{Int32}, vals::Vector{Float32}, idx::Int32, entries::Int32)
    # coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxC
    # println("owner $ic neighbor $ir into $idx")
    if ir in [990000, 999900, 999999] && idx > 6939995
        println("owner: $ic, neighbor: $ir, idx: $idx")
    end
    @inbounds cols[idx] = ic
    @inbounds rows[idx] = ir
    @inbounds vals[idx] += val    # x  
    @inbounds vals[idx+entries] += val  # y
    @inbounds vals[idx+entries+entries] += val  # z
end

function setIndex!(ic::Int32, ir::Int32, val::MVector{3,Float32}, rows::Vector{Int32}, cols::Vector{Int32}, vals::Vector{Float32}, idx::Int32, entries::Int32)
    # coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxC
    @inbounds cols[idx] = ic
    @inbounds rows[idx] = ir
    @inbounds vals[idx] += val[1]    # x  
    @inbounds vals[idx+entries] += val[2]  # y
    @inbounds vals[idx+entries+entries] += val[3]  # z
end

function BatchedFaceBasedAssembly(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets, negOffsets, vals = estimate_data_facebased(input)
    # offsets: the exact index the owner cell is placed in relation to neighbor indices, 
    # negOffsets: where the cell faces start in relation to owner id
    # e.g: row values [1,2,101,10001,1,2,3...]
    #      offsets:   [1               2]
    #      negOffsets:[0               -1] := 1 is already correctly placed, 1 neighbor is smaller than 2 => -1
    # decrease negoffsets to walk from left to right
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        fluxCn = nu[iOwner] * theFace.gDiff
        fluxFn = -fluxCn
        # set diag and upper 
        setIndex!(iOwner, iOwner, fluxCn, rows, cols, vals, offsets[iOwner], entriesNeeded)
        setIndex!(iOwner, iNeighbor, fluxFn, rows, cols, vals, offsets[iOwner] + seenOwner[iOwner], entriesNeeded)
        # increment für symmetry 
        seenOwner[iOwner] += 1
        # set diag and lower
        setIndex!(iNeighbor, iNeighbor, fluxCn, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        setIndex!(iNeighbor, iOwner, fluxFn, rows, cols, vals, offsets[iNeighbor] - negOffsets[iNeighbor], entriesNeeded)
        # increment für symmetry 
        negOffsets[iNeighbor] -= 1
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
            fluxCn = nu[theFace.iOwner] * theFace.gDiff
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            @inbounds fluxVb = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCn
            setIndex!(theFace.iOwner, theFace.iOwner, fluxCn, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
            @inbounds RHS[theFace.iOwner] -= fluxVb[1]
            @inbounds RHS[theFace.iOwner+nCells] -= fluxVb[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= fluxVb[3]
        end
    end
    return rows, cols, vals, RHS
end # function batchedFaceBasedAssembly

lerp(a::MVector{3,Float32}, b::MVector{3,Float32})::MVector{3,Float32} = 0.5 * (a .+ b)


function upwind(ownerVal::MVector{3,Float32}, otherVal::MVector{3,Float32}, Sf::MVector{3,Float32}, out::MVector{3,Float32})
    u_facelocal = lerp(ownerVal, otherVal)
    mdot = dot(u_facelocal, Sf)
    if mdot > 0
        out .= mdot .* ownerVal  # mf ϕC
    end
    out .= mdot .* otherVal   # ̇mf ϕN  (11.35)
end

function centralDifferencing(ownerVal::MVector{3,Float32}, otherVal::MVector{3,Float32}, Sf::MVector{3,Float32}, u_facelocal::MVector{3,Float32}, out::MVector{3,Float32})
    u_facelocal .= lerp(ownerVal, otherVal)
    out .= dot(u_facelocal, Sf) .* u_facelocal   # mdot * ϕf 
end


function DivLapBatchedFaceBasedAssembly(input::LdcMatrixAssemblyInput, interpolation::Function)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    velocity_internal = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = nCells + 2 * mesh.numInteriorFaces
    RHS = zeros(Float32, nCells * 3)
    offsets::Vector{Int32}, negOffsets::Vector{Int32}, vals::Vector{Float32} = estimate_data_facebased(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)
    fluxCn = MVector{3,Float32}(0.0, 0.0, 0.0)
    u_facelocal = MVector{3,Float32}(0.0, 0.0, 0.0)

    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        centralDifferencing(velocity_internal[theFace.iOwner], velocity_internal[theFace.iNeighbor], theFace.Sf, u_facelocal, fluxCn)           ## div(phi, U)      ⟹ Convection
        # fluxCn = nu[iOwner] * theFace.gDiff                                                                                ## laplacian(Γ, U)  ⟹ Diffusion
        fluxFn = -fluxCn
        # set diag and upper 
        setIndex!(iOwner, iOwner, fluxCn, rows, cols, vals, offsets[iOwner], entriesNeeded)
        setIndex!(iOwner, iNeighbor, fluxFn, rows, cols, vals, offsets[iOwner] + seenOwner[iOwner], entriesNeeded)
        # increment für symmetry 
        seenOwner[iOwner] += 1
        # set diag and lower
        setIndex!(iNeighbor, iNeighbor, fluxCn, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
        setIndex!(iNeighbor, iOwner, fluxFn, rows, cols, vals, offsets[iNeighbor] - negOffsets[iNeighbor], entriesNeeded)
        # increment für symmetry 
        negOffsets[iNeighbor] -= 1
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
            # fluxCn = nu[theFace.iOwner] * theFace.gDiff
            @inbounds relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
            centralDifferencing(velocity_internal[theFace.iOwner], velocity_boundary[iBoundary].values[relativeFaceIndex], theFace.Sf, u_facelocal, fluxCn)           ## div(phi, U)      ⟹ Convection
            @inbounds fluxVb = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCn
            setIndex!(theFace.iOwner, theFace.iOwner, fluxCn, rows, cols, vals, offsets[theFace.iOwner], entriesNeeded)
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
function GPU_estimate_data_facebased(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    velocity_internal::Vector{MVector{3,Float32}} = input.U[2].values
    entries = size(mesh.cells)[1] + 2 * mesh.numInteriorFaces
    nCells = size(mesh.cells)[1]
    vals_g = CuArray{Float32}(undef, entries * 3)

    offsets::Vector{Int32} = ones(Int32, nCells)
    gpu_offsets = CuArray{Int32}(undef, nCells)
    # offsets[1] = 1
    for iElement in 2:nCells
        offsets[iElement] += offsets[iElement-1] + mesh.cells[iElement-1].numIntFaces
    end
    # println(vals)
    vals_g[1:nCells] = getindex.(velocity_internal, 1)
    vals_g[entries+1:entries+nCells] = getindex.(velocity_internal, 2)
    vals_g[entries+entries+1:entries+entries+nCells] = getindex.(velocity_internal, 3)
    copyto!(gpu_offsets, offsets)
    offsets = []
    return gpu_offsets, vals_g
end



function GPUFaceBasedAssembly(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    nCells = size(mesh.cells)[1]
    RHS = zeros(Float32, nCells * 3)
    entriesNeeded = size(mesh.cells)[1] + 2 * mesh.numInteriorFaces

    offsets, vals = GPU_estimate_data_facebased(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    seenOwner = ones(Int32, nCells)
    @inbounds for theFace in mesh.faces
        fluxCn = nu * theFace.gDiff
        if theFace.iNeighbor > 0
            iOwner = theFace.iOwner
            iNeighbor = theFace.iNeighbor
            fluxFn = -fluxCn
            # set diag and upper 
            setIndex!(iOwner, iOwner, fluxCn, rows, cols, vals, offsets[iOwner], entriesNeeded)
            setIndex!(iOwner, iNeighbor, fluxFn, rows, cols, vals, offsets[iOwner] + seenOwner[iOwner], entriesNeeded)
            # increment für symmetry 
            seenOwner[iOwner] += 1
            # set diag and lower
            setIndex!(iNeighbor, iNeighbor, fluxCn, rows, cols, vals, offsets[iNeighbor], entriesNeeded)
            setIndex!(iNeighbor, iOwner, fluxFn, rows, cols, vals, offsets[iNeighbor] + seenOwner[iNeighbor], entriesNeeded)
            # increment für symmetry 
            seenOwner[iNeighbor] += 1
        else
            @inbounds iBoundary = theFace.patchIndex
            @inbounds boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = theFace.index - mesh.boundaries[iBoundary].startFace
            fluxVb::Vector{Float32} = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCn
            idx = offsets[theFace.iOwner]
            setIndex!(theFace.iOwner, theFace.iOwner, fluxCn, rows, cols, vals, idx, entriesNeeded)
            @inbounds RHS[theFace.iOwner] -= fluxVb[1]
            @inbounds RHS[theFace.iOwner+nCells] -= fluxVb[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= fluxVb[3]
        end
    end
    return rows, cols, vals
end # function faceBasedAssembly

function facesToGPUarrays(faces)
    numFaces = length(faces)
    iOwners = CuArray{Int32}(undef, numFaces)
    iNeighbors = CuArray{Int32}(undef, numFaces)
    gDiffs = CuArray{Float32}(undef, numFaces)
    relativesO = CuArray{Int32}(undef, numFaces)
    relativesN = CuArray{Int32}(undef, numFaces)

    relativesN[1:numFaces] = [f.relativeToNeighbor for f in faces]
    relativesO[1:numFaces] = [f.relativeToOwner for f in faces]
    iOwners[1:numFaces] = [f.iOwner for f in faces]
    iNeighbors[1:numFaces] = [f.iNeighbor for f in faces]
    gDiffs[1:numFaces] = [f.gDiff for f in faces]
    return iOwners, iNeighbors, gDiffs, relativesO, relativesN
end

function test(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    cells = mesh.cells
    for cell in cells
        ownerIdx = -1
        for iFace in 1:cell.numIntFaces
            iFaceIndex = cell.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if ownerIdx == -1 && theFace.iOwner == cell.index &&theFace.iOwner < theFace.iNeighbor
                theFace.relativeToOwner = 0
                ownerIdx = iFace
                continue
            end
            if cell.index == theFace.iOwner
                theFace.relativeToOwner = iFace - ownerIdx +1 
            else
                theFace.relativeToNeighbor = iFace - ownerIdx 
            end
        end
        if ownerIdx == -1
            ownerIdx = cell.numIntFaces +1
        end
        for iFace in 1:cell.numIntFaces
            iFaceIndex = cell.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if cell.index == theFace.iOwner
                theFace.relativeToOwner = iFace - ownerIdx+1
            else
                theFace.relativeToNeighbor = iFace - ownerIdx 
            end
        end
    end
end

# function kernel_internal_owner(internalFaces::Vector{Face}, offsets::CuArray{Int32}, nu::CuArray{Float32}, seenOwner::CuArray{Int32}, negOffsets::CuArray{Int32}, rows::CuArray{Int32}, cols::CuArray{Int32}, vals::CuArray{Float32})
function kernel_internalFace(iOwners, iNeighbors, gDiffs, offsets, nu, rows, cols, vals, entriesNeeded, relativeToOwner, numInteriorFaces, relativeToNeighbor)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    iOwner = iOwners[iFace]
    iNeighbor = iNeighbors[iFace]
    fluxCn = nu[iOwner] * gDiffs[iFace]
    fluxFn = -fluxCn
    
    idx = offsets[iOwner]
    cols[idx] = iOwner
    rows[idx] = iOwner
    vals[idx] += fluxCn    # x  
    vals[idx+entriesNeeded] += fluxCn  # y
    vals[idx+entriesNeeded+entriesNeeded] += fluxCn  # z

    idx = offsets[iOwner] + relativeToOwner[iFace]
    cols[idx] = iOwner
    rows[idx] = iNeighbor
    vals[idx] += fluxFn    # x  
    vals[idx+entriesNeeded] += fluxFn  # y
    vals[idx+entriesNeeded+entriesNeeded] += fluxFn  # z
    # set diag and lower

    idx = offsets[iNeighbor]
    cols[idx] = iNeighbor
    rows[idx] = iNeighbor
    vals[idx] += fluxCn    # x  
    vals[idx+entriesNeeded] += fluxCn  # y
    vals[idx+entriesNeeded+entriesNeeded] += fluxCn  # z

    idx = offsets[iNeighbor] + relativeToNeighbor[iFace]
    cols[idx] = iNeighbor
    rows[idx] = iOwner
    vals[idx] += fluxFn    # x  
    vals[idx+entriesNeeded] += fluxFn  # y
    vals[idx+entriesNeeded+entriesNeeded] += fluxFn  # z
    return nothing
end


# function kernel_internal_owner(internalFaces::Vector{Face}, offsets::CuArray{Int32}, nu::CuArray{Float32}, seenOwner::CuArray{Int32}, negOffsets::CuArray{Int32}, rows::CuArray{Int32}, cols::CuArray{Int32}, vals::CuArray{Float32})
function kernel_boundaryFace(iOwners, gDiffs, offsets, nu, vals, entriesNeeded, associatedBoundary, velocity_boundary, startFaces, RHS, numInternalFaces, valueOffsets)
    iFace = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @cuprintln("HI")
    iFace = numInternalFaces + iFace
    iOwner = iOwners[iFace]
    fluxCn = nu[iOwner] * gDiffs[iFace]  
    if associatedBoundary[iFace] == 0  # corresponds to type != fixedValue
        return
    end
    relativeFaceIndex = iFace - startFaces[associatedBoundary[iFace]]
    valueIndex = valueOffsets[associatedBoundary[iFace]] + (relativeFaceIndex-1)*3 +1
    fluxVb = velocity_boundary[valueIndex:valueIndex+2] .* -fluxCn 
    idx = offsets[iOwner]
    vals[idx] += fluxCn    # x  
    vals[idx+entriesNeeded] += fluxCn  # y
    vals[idx+entriesNeeded+entriesNeeded] += fluxCn  # z
    RHS[theFace.iOwner] -= fluxVb[1]
    RHS[theFace.iOwner+nCells] -= fluxVb[2]
    RHS[theFace.iOwner+nCells+nCells] -= fluxVb[3]
    return nothing
end


function gpuestimate(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    velocity_internal = input.U[2].values
    entries = size(mesh.cells)[1] + 2 * mesh.numInteriorFaces
    nCells = size(mesh.cells)[1]
    vals_g = CuArray{Float32}(undef, entries * 3)
    offsets = ones(Int32, nCells)
    gpu_offsets = CuArray{Int32}(undef, nCells)
    negOffsets::Vector{Int32} = zeros(Int32, nCells)
    for iElement in 2:nCells
        offsets[iElement] += offsets[iElement-1] + mesh.cells[iElement-1].numIntFaces
    end
    vals_g[1:nCells] = getindex.(velocity_internal, 1)
    vals_g[entries+1:entries+nCells] = getindex.(velocity_internal, 2)
    vals_g[entries+entries+1:entries+entries+nCells] = getindex.(velocity_internal, 3)
    for iElement in 1:nCells
        theElement = mesh.cells[iElement]
        for iFace in 1:theElement.numIntFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor > iElement
                negOffsets[theFace.iNeighbor] += 1
            end
        end
        offsets[iElement] += negOffsets[iElement]  # increase offset
    end
    copyto!(gpu_offsets, offsets)
    offsets = []
    negOffsets = []
    return gpu_offsets, vals_g
end

function uuuh(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    nu_g = CuArray{Float32}(undef, nu.size[1]);
    copyto!(nu_g, nu)
    nCells = size(mesh.cells)[1]
    RHS = CUDA.zeros(Float32, nCells * 3)
    entriesNeeded = size(mesh.cells)[1] + 2 * mesh.numInteriorFaces

    offsets::CuArray{Int32}, vals = gpuestimate(input)
    rows::CuArray{Int32} = CUDA.zeros(Int32, entriesNeeded)
    cols = CUDA.zeros(Int32, entriesNeeded)
    N = length(mesh.numInteriorFaces)
    threads = 256
    blocks  = cld(N, threads)
    println("launching with $threads threads and $blocks blocks")
    test(input)
    iOwners, iNeighbors, gDiffs, relativeToOwners,relativeToNbs  = facesToGPUarrays(mesh.faces);
    @cuda threads=threads blocks=blocks kernel_internalFace(iOwners, iNeighbors, gDiffs, offsets, nu_g, rows, cols, vals, entriesNeeded, relativeToOwners, N, relativeToNbs)
    M = length(mesh.numBoundaryFaces)
    threads = 256
    blocks  = cld(M, threads)
    faceBoundaryMapping, startFaces, velocity_boundary, valueOffsets = associatedBoundaries(input)
    @cuda threads=1 blocks=1 kernel_boundaryFace(iOwners, gDiffs, offsets, nu_g, vals, entriesNeeded, faceBoundaryMapping, velocity_boundary, startFaces, RHS, N, valueoffsets)
    return rows, cols, vals
end

function associatedBoundaries(input::LdcMatrixAssemblyInput)
    mapping = zeros(Int32, input.mesh.numBoundaryFaces)
    startFaces = zeros(Int32, length(input.mesh.boundaries))
    for iBoundary in eachindex(input.mesh.boundaries)
        theBoundary = input.mesh.boundaries[iBoundary]
        if input.U[1][iBoundary].type != "fixedValue"
            startFaces[iBoundary] = -1
            continue
        end 
        startFace = theBoundary.startFace + 1
        startFaces[iBoundary] = startFace
        endFace = startFace + theBoundary.nFaces
        mapping[startFace-input.mesh.numInteriorFaces:endFace-1-input.mesh.numInteriorFaces] .= theBoundary.index
    end
    velocities = [b.values for b in input.U[1]]
    vlength = sum(velocities.size[1] .*[v.size[1] for v in velocities])*3
    newvels = zeros(Float32, vlength)
    i = 1
    for v in velocities
        for vv in v
            if i < 10
                println("setting newvels[$i] to $vv")
            end
            newvels[i]   = vv[1] 
            newvels[i+1] = vv[2]
            newvels[i+2] = vv[3]
            i += 3
        end
    end
    valueOffsets = [1]
    for b in [v.size[1] for v in velocities[2:end]]
        push!(valueOffsets, b)
    end
    return CuArray(mapping), CuArray(startFaces), CuArray(newvels), CuArray(valueoffsets)
end