using LinearAlgebra
using StaticArrays
using ElasticArrays
using .Threads
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
    inGroups::Tuple{Int,String}
    nFaces::Int
    startFace::Int
end # struct Boundary

mutable struct Cell
    index::Int
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
    println("Reading Points..")
    nodes = readPointsFile(polymeshDir)
    println("Reading Owners..")
    owner = readOwnersFile(polymeshDir)
    println("Reading Faces..")
    faces = readFacesFile(polymeshDir, owner)
    println("Reading Neighbors..")
    numNeighbors, faces = readNeighborsFile(polymeshDir, faces)
    numCells = maximum(owner)
    println("Reading Boundaries..")
    boundaries = readBoundaryFile(polymeshDir)
    println("Constructing Cells..")
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
        s = replace(l[iBound], r"\s+" => " ")
        s = replace(s, ";" => "")
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

function constructCells(nodes::MVector, boundaries::Vector{Boundary}, faces::Vector{Face}, numCells::Int, numInteriorFaces::Int)::Mesh
    cells = [
        Cell(
            index,
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
            println("settings numinteriors to $numInteriors")
            numInteriors = interiorFace - 1
            break
        end
        push!(cells[iOwner].iFaces, faces[interiorFace].index)
        push!(cells[iOwner].iNeighbors, iNeighbor)
        push!(cells[iOwner].faceSigns, 1)
        push!(cells[iNeighbor].iFaces, faces[interiorFace].index)
        push!(cells[iNeighbor].iNeighbors, iOwner)
        push!(cells[iNeighbor].faceSigns, -1)
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
        println("isuniform")
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
    println("Reading $(path)")
    if !isfile(path)
        throw(CaseDirError("Field file '$(path)' does not exist."))
    end
    file = read(path, String)
    variable = match(r"\w+\s*\[[\s\d-]+\]\s*([\.\d\-e]+);", file)
    println("Variable: $(variable)")
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

using SparseArrays
using StaticArrays
using ProgressBars

function cellBasedAssembly(input::LdcMatrixAssemblyInput)::Tuple{Matrix{Float32},Vector{Float32}}
    mesh = input.mesh
    source = input.source
    diffusionCoeff = input.diffusionCoeff # Γ
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    coeffMatrix = Matrix(zeros(nCells, nCells))
    for (iElement, theElement) in enumerate(mesh.cells)
        # bC = Q_C*V_C - bigsum(FluxVf)
        RHS[iElement] = source[iElement] * theElement.volume  # S_u
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                # FluxCn = Γ_n * gdiff_n  (p.215)
                fluxCn = diffusionCoeff[iFaceIndex] * theFace.gDiff
                fluxFn = -fluxCn
                coeffMatrix[iElement, theElement.iNeighbors[iFace]] = fluxFn
                # accumulate neighbors into diag 
                coeffMatrix[iElement, iElement] += fluxCn
            else
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                boundaryType = boundaryFields[iBoundary].type
                if boundaryType == "fixedValue"
                    fluxCb = diffusionCoeff[iFaceIndex] * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb = -fluxCb * boundaryFields[iBoundary].values[relativeFaceIndex]
                    coeffMatrix[iElement, iElement] += fluxCb
                    RHS[iElement] -= fluxVb
                end
            end
        end
    end
    return coeffMatrix, RHS
end # function cellBasedAssembly



function cellBasedAssemblySparseMatrix(input::MatrixAssemblyInput)::Tuple{SparseMatrixCSC{Float32,Int64},SparseVector{Float32,Int64}}
    mesh = input.mesh
    source = input.source
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = spzeros(nCells)
    coeffMatrix = spzeros(nCells, nCells)
    for (iElement, theElement) in enumerate(mesh.cells)
        RHS[iElement] = source[iElement] * theElement.volume
        diag = 0.0
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                fluxCn = diffusionCoeff[iFaceIndex] * theFace.gDiff
                fluxFn = -fluxCn
                coeffMatrix[iElement, theElement.iNeighbors[iFace]] = fluxFn
                diag += fluxCn
            else
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                boundaryType = boundaryFields[iBoundary].type
                if boundaryType == "fixedValue"
                    fluxCb = diffusionCoeff[iFaceIndex] * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb = -fluxCb * boundaryFields[iBoundary].values[relativeFaceIndex]
                    diag += fluxCb
                    RHS[iElement] -= fluxVb
                end
            end
        end
        coeffMatrix[iElement, iElement] = diag
    end
    return coeffMatrix, RHS
end # function cellBasedAssemblySparseMatrix

function cellBasedAssemblySparseMultiVectorPrealloc(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    source = input.source
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    rows = zeros(nCells * nCells)
    cols = zeros(nCells * nCells)
    vals::Vector{Float32} = zeros(nCells * nCells)
    idx = 1
    for (iElement, theElement) in enumerate(mesh.cells)
        RHS[iElement] = source[iElement] * theElement.volume
        diag = 0.0
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                fluxCn = diffusionCoeff[iFaceIndex] * theFace.gDiff
                fluxFn = -fluxCn
                rows[idx] = iElement
                cols[idx] = theElement.iNeighbors[iFace]
                vals[idx] = fluxFn
                idx += 1
                diag += fluxCn
            else
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                boundaryType = boundaryFields[iBoundary].type
                if boundaryType == "fixedValue"
                    fluxCb = diffusionCoeff[iFaceIndex] * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb = -fluxCb * boundaryFields[iBoundary].values[relativeFaceIndex]
                    diag += fluxCb
                    RHS[iElement] -= fluxVb
                end
            end
        end
        rows[idx] = iElement
        cols[idx] = iElement
        vals[idx] = diag
        idx += 1
    end
    idx -= 1
    return SparseArrays.sparse(
        rows[1:idx],
        cols[1:idx],
        vals[1:idx],
    ), RHS
end # function cellBasedAssemblySparseMultiVectorPrealloc

function cellBasedAssemblySparseMultiVectorPush(input::MatrixAssemblyInput)::Tuple{SparseMatrixCSC{Float32,Int64},Vector{Float32}}
    mesh = input.mesh
    source = input.source
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    rows = []
    cols = []
    vals::Vector{Float32} = []
    for (iElement, theElement) in enumerate(mesh.cells)
        RHS[iElement] = source[iElement] * theElement.volume
        diag = 0.0
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                fluxCn = diffusionCoeff[iFaceIndex] * theFace.gDiff
                fluxFn = -fluxCn
                push!(rows, iElement)
                push!(cols, theElement.iNeighbors[iFace])
                push!(vals, fluxFn)
                diag += fluxCn
            else
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                boundaryType = boundaryFields[iBoundary].type
                if boundaryType == "fixedValue"
                    fluxCb = diffusionCoeff[iFaceIndex] * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb = -fluxCb * boundaryFields[iBoundary].values[relativeFaceIndex]
                    diag += fluxCb
                    RHS[iElement] -= fluxVb
                end
            end
        end
        push!(rows, iElement)
        push!(cols, iElement)
        push!(vals, diag)
    end
    return SparseArrays.sparse(
        rows, cols, vals
    ), RHS
end # function cellBasedAssemblySparseMultiVectorPush

function faceBasedAssembly(input::MatrixAssemblyInput)::Tuple{Matrix{Float32},Vector{Float32}}
    mesh = input.mesh
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    coeffMatrix = Matrix(zeros(nCells, nCells))
    for (iFace, theFace) in enumerate(mesh.faces)
        fluxC = diffusionCoeff[iFace] * theFace.gDiff
        if theFace.iNeighbor != -1
            fluxFn = -fluxC
            coeffMatrix[theFace.iOwner, theFace.iNeighbor] = fluxFn
            coeffMatrix[theFace.iNeighbor, theFace.iOwner] = fluxFn
            coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxC
            coeffMatrix[theFace.iNeighbor, theFace.iNeighbor] += fluxC
        else
            iBoundary = mesh.faces[iFace].patchIndex
            boundaryType = boundaryFields[iBoundary].type
            if boundaryType == "fixedValue"
                relativeFaceIndex = iFace - mesh.boundaries[iBoundary].startFace
                fluxVb = -fluxC * boundaryFields[iBoundary].values[relativeFaceIndex]
                coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxC
                RHS[theFace.iOwner] -= fluxVb
            end
        end
    end
    return coeffMatrix, RHS
end # function faceBasedAssembly


function batchedFaceBasedAssembly(input::MatrixAssemblyInput)::Tuple{Matrix{Float32},Vector{Float32}}
    mesh = input.mesh
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    coeffMatrix = Matrix(zeros(nCells, nCells))
    nInteriorFaces = mesh.numInteriorFaces
    for (iFace, theFace) in enumerate(mesh.faces[1:nInteriorFaces])
        fluxCn = diffusionCoeff[iFace] * theFace.gDiff
        fluxFn = -fluxCn
        coeffMatrix[theFace.iOwner, theFace.iNeighbor] = fluxFn
        coeffMatrix[theFace.iNeighbor, theFace.iOwner] = fluxFn

        coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxCn
        coeffMatrix[theFace.iNeighbor, theFace.iNeighbor] += fluxCn
    end
    for (iBoundary, theBoundary) in enumerate(mesh.boundaries)
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        if boundaryFields[iBoundary].type == "fixedValue"
            for iFace in startFace:endFace-1
                theFace = mesh.faces[iFace]
                fluxCb = diffusionCoeff[iFace] * theFace.gDiff
                relativeFaceIndex = iFace - mesh.boundaries[theFace.patchIndex].startFace
                fluxVb = -fluxCb * boundaryFields[iBoundary].values[relativeFaceIndex]
                RHS[theFace.iOwner] -= fluxVb
                coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxCb
            end
        end
    end
    return coeffMatrix, RHS
end # function batchedFaceBasedAssembly

function batchedFaceBasedAssemblySparseMatrix(input::MatrixAssemblyInput)::Tuple{SparseMatrixCSC{Float32,Int64},SparseVector{Float32,Int64}}
    mesh = input.mesh
    diffusionCoeff = input.diffusionCoeff
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    RHS = spzeros(nCells)
    coeffMatrix = spzeros(nCells, nCells)
    nInteriorFaces = mesh.numInteriorFaces
    for (iFace, theFace) in enumerate(mesh.faces[1:nInteriorFaces])
        fluxCn = diffusionCoeff[iFace] * theFace.gDiff
        fluxFn = -fluxCn
        coeffMatrix[theFace.iOwner, theFace.iNeighbor] = fluxFn
        coeffMatrix[theFace.iNeighbor, theFace.iOwner] = fluxFn

        coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxCn
        coeffMatrix[theFace.iNeighbor, theFace.iNeighbor] += fluxCn
    end
    for (iBoundary, theBoundary) in enumerate(mesh.boundaries)
        startFace = theBoundary.startFace + 1
        endFace = startFace + theBoundary.nFaces
        if boundaryFields[iBoundary].type == "fixedValue"
            for iFace in startFace:endFace-1
                theFace = mesh.faces[iFace]
                fluxCb = diffusionCoeff[iFace] * theFace.gDiff
                relativeFaceIndex = iFace - mesh.boundaries[theFace.patchIndex].startFace
                fluxVb = -fluxCb * boundaryFields[iBoundary].values[relavalstiveFaceIndex]
                RHS[theFace.iOwner] -= fluxVb
                coeffMatrix[theFace.iOwner, theFace.iOwner] += fluxCb
            end
        end
    end
    return coeffMatrix, RHS
end # function batchedFaceBasedAssemblySparseMatrix

function genericCellBasedAssemblySparseMatrix(input::GenericMatrixAssemblyInput)::Tuple{Vector{SparseMatrixCSC{Float32,Int64}},Vector{Vector{Float32}}}
    mesh = input.mesh
    variables = input.variables
    boundaryFields = input.boundaryFields
    nCells = size(mesh.cells)[1]
    mappings = input.mappings
    numVariables = length(mappings)
    coeffMatrices = [spzeros(nCells, nCells) for _ in 1:numVariables]
    RHSs = input.sources != [] ? input.sources : [zeros(nCells) for _ in 1:numVariables]
    for (iElement, theElement) in enumerate(mesh.cells)
        for iVar in 1:numVariables
            RHSs[iVar][iElement] *= theElement.volume  # S_u
        end
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            for (iVariable, variable) in enumerate(variables)
                if theFace.iNeighbor != -1
                    fluxCn = variable[iFaceIndex] * theFace.gDiff
                    fluxFn = -fluxCn
                    coeffMatrices[iVariable][iElement, theElement.iNeighbors[iFace]] = fluxFn
                    coeffMatrices[iVariable][iElement, iElement] += fluxCn
                else
                    iBoundary = mesh.faces[iFaceIndex].patchIndex
                    boundaryType = boundaryFields[iVariable][iBoundary].type
                    if boundaryType == "fixedValue"
                        fluxCb = variable[iFaceIndex] * theFace.gDiff
                        relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                        fluxVb = -fluxCb * boundaryFields[iVariable][iBoundary].values[relativeFaceIndex]
                        coeffMatrices[iVariable][iElement, iElement] += fluxCb
                        RHSs[iVariable][iElement] -= fluxVb
                    end
                end
            end
        end
    end
    return coeffMatrices, RHSs
end # function genericCellBasedAssemblySparseMatrix


function ScalarAssembly(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    velocity_internal = input.U[2].values
    nCells = size(mesh.cells)[1]
    RHS = spzeros(nCells, 3)  # [ux,uy,uz]
    rows = []
    cols = []
    vals::Vector{Float32} = []
    for (iElement, theElement) in enumerate(mesh.cells)
        diag = velocity_internal[iElement]
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1
                fluxCn = nu * theFace.gDiff
                fluxFn = -fluxCn
                push!(rows, iElement)
                push!(cols, theElement.iNeighbors[iFace])
                push!(vals, fluxFn)
                diag += fluxCn
            else
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                boundaryType = velocity_boundary[iBoundary].type
                if boundaryType == "fixedValue"
                    fluxCb = nu * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCb
                    RHS[iElement, :] .-= fluxVb
                    diag .+= fluxCb
                end
            end
        end
        push!(rows, iElement)
        push!(cols, iElement)
        push!(vals, diag)
    end

    return SparseArrays.sparse(
        rows,
        cols,
        vals,
    ), RHS
end # function cellBasedAssemblySparseMultiVectorPrealloc

function VectorAssembly(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    velocity_internal = input.U[2].values
    nCells = size(mesh.cells)[1]
    RHS = spzeros(nCells, 3)  # [ux,uy,uz]
    rows = ElasticArray{Int}(undef, 1000000)
    cols = ElasticArray{Int}(undef, 1000000)
    vals = ElasticArray{MVector{3,Float32}}(undef, 3, 1000000)

    # valsx = ElasticArray{Float32}(undef, 1000000)
    # valsy = ElasticArray{Float32}(undef, 1000000)
    # valsz = ElasticArray{Float32}(undef, 1000000)


    # vals::Vector{MVector{3, Float32}} = []
    idx = 1
    println("First stage:")
    @time begin
        for (iElement, theElement) in enumerate(mesh.cells)
            diag = velocity_internal[iElement]
            for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
                theFace = mesh.faces[iFaceIndex]
                if theFace.iNeighbor > 0
                    fluxCn = nu * theFace.gDiff
                    fluxFn = -fluxCn
                    # println("$idx  >  $(size(vals)[2])")
                    if idx > size(vals)[1]
                        newSize = floor(Int, min(size(vals)[1] * 1.5, nCells * nCells))
                        # println("newSize: $newSize")
                        # resize!(valsx, newSize)
                        # resize!(valsy, newSize)
                        # resize!(valsz, newSize)
                        resize!(vals, newSize)
                        resize!(rows, newSize)
                        resize!(cols, newSize)
                    end
                    @inbounds rows[idx] = iElement
                    @inbounds cols[idx] = theElement.iNeighbors[iFace]
                    @inbounds vals[idx] = MVector(fluxFn, fluxFn, fluxFn)
                    # valsx[idx] = fluxFn
                    # valsy[idx] = fluxFn
                    # valsz[idx] = fluxFn

                    idx += 1
                    # push!(rows, iElement)
                    # push!(cols, theElement.iNeighbors[iFace])
                    # push!(vals, MVector(fluxFn, fluxFn, fluxFn))
                    diag .+= fluxCn
                else
                    iBoundary = mesh.faces[iFaceIndex].patchIndex
                    boundaryType = velocity_boundary[iBoundary].type
                    if boundaryType == "fixedValue"
                        fluxCb = nu * theFace.gDiff
                        relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                        fluxVb = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCb
                        RHS[iElement, :] .-= fluxVb
                        diag .+= fluxCb
                    end
                end
            end
            if idx > size(vals)[1]
                newSize = floor(Int, min(size(vals)[1] * 1.5, nCells))
                resize!(vals, newSize)
                # resize!(valsx, newSize)
                # resize!(valsy, newSize)
                # resize!(valsz, newSize)
                resize!(rows, newSize)
                resize!(cols, newSize)
            end
            @inbounds rows[idx] = iElement
            @inbounds cols[idx] = iElement
            @inbounds valsx[idx] = diag
            # valsx[idx] = diag[1]
            # valsy[idx] = diag[2]
            # valsz[idx] = diag[3]

            idx += 1

            # push!(rows, iElement)
            # push!(cols, iElement)
            # push!(vals, diag)
        end
    end
    println("Second stage:")

    idx -= 1
    @time begin
        mat = SparseArrays.sparse(
            rows[1:idx],
            cols[1:idx],
            vals[1:idx]
        )
    end

end # function cellBasedAssemblySparseMultiVectorPrealloc

function ThreadedAssembly(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nCells = size(mesh.cells)[1]
    RHSx::Vector{Float32} = zeros(nCells)  # [ux,uy,uz]
    RHSy::Vector{Float32} = zeros(nCells)  # [ux,uy,uz]
    RHSz::Vector{Float32} = zeros(nCells)  # [ux,uy,uz]
    entriesNeeded, offsets = estimate_data(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    valsx = Vector{Float32}(undef, entriesNeeded)
    valsy = Vector{Float32}(undef, entriesNeeded)
    valsz = Vector{Float32}(undef, entriesNeeded)
    sizehint!(valsx, entriesNeeded)
    sizehint!(valsy, entriesNeeded)
    sizehint!(valsz, entriesNeeded)
    chunks = Iterators.partition(1:nCells, nCells ÷ nthreads())
    chunks = [c for c in chunks]
    tasks = map(chunks) do chunk
        # @spawn begin
        #     for iElement in chunk
        #         theElement = mesh.cells[iElement]
        #         numFaces = size(theElement.iFaces)[1]
        #         @inbounds diagx::Float32 = velocity_internal[iElement][1]
        #         @inbounds diagy::Float32 = velocity_internal[iElement][2]
        #         @inbounds diagz::Float32 = velocity_internal[iElement][3]
        #         @inbounds for iFace in 1:numFaces
        #             @inbounds iFaceIndex = theElement.iFaces[iFace]
        #             @inbounds theFace = mesh.faces[iFaceIndex]
        #             if theFace.iNeighbor > 0
        #                 fluxCn = nu * theFace.gDiff
        #                 fluxFn = -fluxCn
        #                 idx = offsets[iElement] + iFace
        #                 @inbounds cols[idx] = iElement
        #                 @inbounds rows[idx] = theElement.iNeighbors[iFace]
        #                 @inbounds valsx[idx] = fluxFn
        #                 @inbounds valsy[idx] = fluxFn
        #                 @inbounds valsz[idx] = fluxFn
        #                 diagx += fluxCn
        #                 diagy += fluxCn
        #                 diagz += fluxCn
        #             else
        #                 @inbounds iBoundary = mesh.faces[iFaceIndex].patchIndex
        #                 @inbounds boundaryType = velocity_boundary[iBoundary].type
        #                 if boundaryType == "fixedValue"
        #                     fluxCb = nu * theFace.gDiff
        #                     relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
        #                     fluxVb::Vector{Float32} = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCb
        #                     RHSx[iElement] -= fluxVb[1]
        #                     RHSy[iElement] -= fluxVb[2]
        #                     RHSz[iElement] -= fluxVb[3]
        #                     diagx += fluxCb
        #                     diagy += fluxCb
        #                     diagz += fluxCb
        #                 end
        #             end
        #         end
        #         idx = offsets[iElement]
        #         @inbounds cols[idx] = iElement
        #         @inbounds rows[idx] = iElement
        #         @inbounds valsx[idx] = diagx
        #         @inbounds valsy[idx] = diagy
        #         @inbounds valsz[idx] = diagz
        #         ## this demolishes performance in multithreaded
        #         # range = offsets[iElement]:(offsets[iElement+1]-1)
        #         # p = sortperm(rows[range])
        #         # @inbounds rows[range] .= rows[range][p]
        #         # @inbounds valsx[range] .= valsx[range][p]
        #         # @inbounds valsy[range] .= valsy[range][p]
        #         # @inbounds valsz[range] .= valsz[range][p]
        #     end
        # end
        @spawn helper(chunk, input, RHSx, RHSy, RHSz, rows, cols, offsets, valsx, valsy, valsz)
    end
    fetch.(tasks)
    mats = Vector{SparseMatrixCSC}(undef, 3)
    for (i, dim) in enumerate([valsx, valsy, valsz])
        @spawn begin
            mats[i] = SparseArrays.sparse(rows, cols, dim)
        end
    end
    while !isassigned(mats, 1) && isassigned(mats, 2) && isassigned(mats, 3)
        continue
    end
    rows = []
    cols = []
    valsx = []
    valsy = []
    valsz = []
    return mats
end

function helper(chunk::UnitRange, input::LdcMatrixAssemblyInput, RHSx::Vector{Float32}, RHSy::Vector{Float32}, RHSz::Vector{Float32}, rows::Vector{Int32}, cols::Vector{Int32}, offsets, valsx::Vector, valsy::Vector, valsz::Vector)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    velocity_internal = input.U[2].values
    for iElement in chunk
        theElement = mesh.cells[iElement]
        numFaces = size(theElement.iFaces)[1]
        @inbounds diagx::Float32 = velocity_internal[iElement][1]
        @inbounds diagy::Float32 = velocity_internal[iElement][2]
        @inbounds diagz::Float32 = velocity_internal[iElement][3]
        @inbounds for iFace in 1:numFaces
            @inbounds iFaceIndex = theElement.iFaces[iFace]
            @inbounds theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor > 0
                fluxCn = nu * theFace.gDiff
                fluxFn = -fluxCn
                idx = offsets[iElement] + iFace
                @inbounds cols[idx] = iElement
                @inbounds rows[idx] = theElement.iNeighbors[iFace]
                @inbounds valsx[idx] = fluxFn
                @inbounds valsy[idx] = fluxFn
                @inbounds valsz[idx] = fluxFn
                diagx += fluxCn
                diagy += fluxCn
                diagz += fluxCn
            else
                @inbounds iBoundary = mesh.faces[iFaceIndex].patchIndex
                @inbounds boundaryType = velocity_boundary[iBoundary].type
                if boundaryType == "fixedValue"
                    fluxCb = nu * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb::Vector{Float32} = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCb
                    RHSx[iElement] -= fluxVb[1]
                    RHSy[iElement] -= fluxVb[2]
                    RHSz[iElement] -= fluxVb[3]
                    diagx += fluxCb
                    diagy += fluxCb
                    diagz += fluxCb
                end
            end
        end
        idx = offsets[iElement]
        @inbounds cols[idx] = iElement
        @inbounds rows[idx] = iElement
        @inbounds valsx[idx] = diagx
        @inbounds valsy[idx] = diagy
        @inbounds valsz[idx] = diagz
    end
end

function VectorAssemblyMulti(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    velocity_internal = input.U[2].values
    nCells = size(mesh.cells)[1]
    RHSx = zeros(Float32, nCells)
    RHSy = zeros(Float32, nCells)
    RHSz = zeros(Float32, nCells)
    entriesNeeded, offsets = estimate_data(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    valsx = Vector{Float32}(undef, entriesNeeded)
    valsy = Vector{Float32}(undef, entriesNeeded)
    valsz = Vector{Float32}(undef, entriesNeeded)

    @threads for iElement = 1:nCells
        theElement = mesh.cells[iElement]
        numFaces = size(theElement.iFaces)[1]
        @inbounds diagx::Float32 = velocity_internal[iElement][1]
        @inbounds diagy::Float32 = velocity_internal[iElement][2]
        @inbounds diagz::Float32 = velocity_internal[iElement][3]
        for iFace in 1:numFaces
            @inbounds iFaceIndex = theElement.iFaces[iFace]
            @inbounds theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor > 0
                fluxCn = nu * theFace.gDiff
                fluxFn = -fluxCn
                idx = offsets[iElement] + iFace
                @inbounds cols[idx] = iElement
                @inbounds rows[idx] = theElement.iNeighbors[iFace]
                @inbounds valsx[idx] = fluxFn
                @inbounds valsy[idx] = fluxFn
                @inbounds valsz[idx] = fluxFn
                diagx += fluxCn
                diagy += fluxCn
                diagz += fluxCn
            else
                @inbounds iBoundary = mesh.faces[iFaceIndex].patchIndex
                @inbounds boundaryType = velocity_boundary[iBoundary].type
                if boundaryType == "fixedValue"
                    fluxCb = nu * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    fluxVb::Vector{Float32} = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCb
                    @inbounds RHSx[iElement] -= fluxVb[1]
                    @inbounds RHSy[iElement] -= fluxVb[2]
                    @inbounds RHSz[iElement] -= fluxVb[3]
                    diagx += fluxCb
                    diagy += fluxCb
                    diagz += fluxCb
                end
            end
        end
        idx = offsets[iElement]
        @inbounds cols[idx] = iElement
        @inbounds rows[idx] = iElement
        @inbounds valsx[idx] = diagx
        @inbounds valsy[idx] = diagy
        @inbounds valsz[idx] = diagz
        # p = sortperm(rows[range])
        # @inbounds rows[range] .= rows[range][p]
        # @inbounds valsx[range] .= valsx[range][p]
        # @inbounds valsy[range] .= valsy[range][p]
        # @inbounds valsz[range] .= valsz[range][p]
    end
    return SparseArrays.sparse(rows, cols, valsx), SparseArrays.sparse(rows, cols, valsy), SparseArrays.sparse(rows, cols, valsz)
    # return SparseMatrixCSC(nCells, nCells, offsets, rows, valsx), SparseMatrixCSC(nCells, nCells, offsets, rows, valsy), SparseMatrixCSC(nCells, nCells, offsets, rows, valsz)
end # function cellBasedAssemblySparseMultiVectorPrealloc


function estimate_data(input::LdcMatrixAssemblyInput)
    mesh = input.mesh
    e2 = size(mesh.cells)[1]
    nCells = size(mesh.cells)[1]

    offsets::Vector{Int32} = ones(e2 + 1)
    for face in mesh.faces
        if face.iNeighbor > 0
            e2 += 2
        end
    end
    offsets[1] = 1
    for iElement in 2:nCells
        faces = mesh.cells[iElement-1].iFaces
        nonzero = size(filter(f -> f.iNeighbor > 0, mesh.faces[faces]))[1]
        offsets[iElement] += offsets[iElement-1] + nonzero
    end
    offsets[end] = e2 + 1
    return e2, offsets
end

function bench_gc(input::LdcMatrixAssemblyInput)
    for _ in 1:10
        GC.gc()
        @time ThreadedAssembly(input)
    end
end

function bench_nogc(input::LdcMatrixAssemblyInput)
    for _ in 1:10
        @time ThreadedAssembly(input)
    end
end

