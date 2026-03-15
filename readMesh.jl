include("classes.jl")
using LinearAlgebra
using StaticArrays

function readOpenFoamMesh(P::Type{<:AbstractFloat}, caseDir::String)::Mesh
    if !isdir(caseDir)
        throw(CaseDirError("Case Directory '$(caseDir)' does not exist"))
    end
    polymeshDir = joinpath(caseDir, "constant/polyMesh")
    if !isdir(polymeshDir)
        throw(CaseDirError("PolyMesh Directory '$(caseDir)' does not exist"))
    end
    # println("Reading Points..")
    nodes = readPointsFile(P, polymeshDir)
    # println("Reading Owners..")
    owner = readOwnersFile(polymeshDir)
    # println("Reading Faces..")
    faces = readFacesFile(P, polymeshDir, owner)
    # println("Reading Neighbors..")
    numNeighbors, faces = readNeighborsFile(polymeshDir, faces)
    numCells = maximum(owner)
    # println("Reading Boundaries..")
    boundaries = readBoundaryFile(polymeshDir)
    # println("Constructing Cells..")
    mesh = constructCells(P, nodes, boundaries, faces, numCells, numNeighbors)
    # mesh = setupNodeConnectivities(mesh)
    mesh = processOpenFoamMesh(mesh, faces)
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

function readPointsFile(P::Type{<:AbstractFloat}, polyMeshDir::String)::Vector{SVector{3, P}}
    pointsFileName = joinpath(polyMeshDir, "points")
    if !isfile(pointsFileName)
        throw(CaseDirError("Points file '$(pointsFileName)' does not exist."))
    end
    nNodes, start = getAmountAndStart(pointsFileName)
    centroids = Vector{SVector{3,P}}(undef, nNodes)
    i = 1
    for (j, line) in enumerate(eachline(pointsFileName))
        if j < start
            continue
        end
        if line == ")"
            break
        end
        s = split((line[2:(end-1)]), " ")
        centroids[i] = SVector{3,P}(tryparse.(P, s))
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

function readFacesFile(P::Type{<:AbstractFloat}, polyMeshDir::String, owners::Vector{Int32})::Vector{TmpFace}
    facesFileName = joinpath(polyMeshDir, "faces")
    if !isfile(facesFileName)
        throw(CaseDirError("Faces file '$(facesFileName)' does not exist."))
    end
    nFaces, start = getAmountAndStart(facesFileName)
    faces = Vector{TmpFace}(undef, nFaces)
    i::Int32 = 1
    for (j, line) in enumerate(eachline(facesFileName))
        if j < start
            continue
        end
        if line == ")"
            break
        end
        faces[i] = TmpFace(
            i,
            parse.(Int32, split(line[3:(end-1)], " ")) .+ one(Int32),
            owners[i],
            -one(Int32),
            # MVector{3,P}(0.0, 0.0, 0.0),
            # MVector{3,P}(0.0, 0.0, 0.0),
            # zero(P),
            # zero(P),
            # -one(Int32),
            # zero(Int32),
            # zero(Int32),
            # -one(Int32)
        )
        i += 1
    end
    return faces
end # function readPointsFile


function readNeighborsFile(polyMeshDir::String, faces::Vector{TmpFace})::Tuple{Int32,Vector{TmpFace}}
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

function constructCells(P::Type{<:AbstractFloat}, nodes::Vector, boundaries::Vector{Boundary}, faces::Vector{TmpFace}, numCells::Int32, numInteriorFaces::Int32)::Mesh{P}
    cells = [
        Cell(
            Int32(index),
            zero(Int32),
            zero(P),
            Int32[],
            Int32[],
            Int32[],
            MVector{3,P}(0.0, 0.0, 0.0),
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
        cells[iOwner].nInternalFaces += 1
        cells[iNeighbor].nInternalFaces += 1
    end
    for boundaryFace in (numInteriors+1):length(faces)
        owner = faces[boundaryFace].iOwner
        push!(cells[owner].iFaces, boundaryFace)
        push!(cells[owner].faceSigns, 1)
    end

    # for cell in cells
    #     cell.numNeighbors = length(cell.iNeighbors)
    # end
    numBoundaryCells = length(faces) - numInteriors
    numBoundaryFaces = length(faces) - numInteriors
    return Mesh{P}(nodes, [], boundaries, numCells, cells, numInteriors, numBoundaryCells, numBoundaryFaces)
end # function constructCells

function setupNodeConnectivities(mesh::Mesh)::Mesh
    for iFace in 1:length(mesh.faces)
        iNodes = mesh.faces[iFace].iNodes
        nNodes = length(iNodes)
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



function processOpenFoamMesh(mesh::Mesh{P}, tmpFaces::Vector{TmpFace})::Mesh{P} where {P<:AbstractFloat}
    mesh = processBasicFaceGeometry(mesh, tmpFaces)
    mesh = computeElementVolumeAndCentroid(mesh)
    mesh = processSecondaryFaceGeometry(mesh)
    # mesh = sortBoundaryNodesFromInteriorNodes(mesh)
    mesh = labelBoundaryFaces(mesh)
    return mesh
end # function processOpenFoamMesh

function magnitude(vector::MVector{3,P})::P where {P<:AbstractFloat}
    d = dot(vector, vector)
    return sqrt(d)
end # function magnitude

function processBasicFaceGeometry(mesh::Mesh{P}, tmpFaces::Vector{TmpFace})::Mesh{P} where {P<:AbstractFloat}
    faces = Vector{Face}(undef, length(tmpFaces))
    for face::TmpFace in tmpFaces
        # special case: triangle
        centroid = MVector{3,P}(zeros(P, 3))
        Sf = MVector{3,P}(zeros(P, 3))
        area = zero(P)
        if length(face.iNodes) == 3
            # sum x,y,z and divide by 3
            triangleNodes = [mesh.nodes[i] for i in face.iNodes]
            centroid .= sum(triangleNodes) / 3
            Sf .= 0.5 * cross(triangleNodes[2] - triangleNodes[1], triangleNodes[3] - triangleNodes[1])
            area = magnitude(Sf)
        else # general case, polygon is not a triangle
            nodes = [mesh.nodes[i] for i in face.iNodes]
            center = sum(nodes) / length(nodes)
            # Using the center to compute the area and centroid of virtual
            # triangles based on the center and the face nodes
            triangleNode1 = center
            triangleNode3 = MVector{3,P}(0.0, 0.0, 0.0)
            for (iNodeIndex, iNode) in enumerate(face.iNodes)
                if iNodeIndex < length(face.iNodes)
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
                centroid .+= localArea * localCentroid
                Sf += localSf
            end
            area = magnitude(Sf)
            # Compute centroid of the polygon
            centroid ./= area
        end
        faces[face.index] = Face{P}(
            face.index,
            face.iOwner,
            face.iNeighbor,
            centroid,
            Sf,
            area,
            zero(P),
            -one(Int32),
            zero(Int32),
            zero(Int32),
            -one(Int32)
        )
    end
    mesh.faces = faces
    return mesh
end # function processBasicFaceGeometry

function computeElementVolumeAndCentroid(mesh::Mesh{P})::Mesh{P} where {P<:AbstractFloat}
    for iElement in 1:(length(mesh.cells))
        iFaces = mesh.cells[iElement].iFaces
        elementCenter = MVector{3,P}(0.0, 0.0, 0.0)
        for iFace in iFaces
            elementCenter .+= mesh.faces[iFace].centroid
        end
        elementCenter ./= length(iFaces)
        localVolumeCentroidSum = zeros(3)
        localVolumeSum = 0.0
        for iFace in eachindex(iFaces)
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

function processSecondaryFaceGeometry(mesh::Mesh{P})::Mesh{P} where {P<:AbstractFloat}
    # Loop over Interior faces
    eCN = MVector{3,P}(0.0, 0.0, 0.0)
    CN = MVector{3,P}(0.0, 0.0, 0.0)

    for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        # Compute unit surtheFace normal vector
        # nf = theFace.Sf ./ theFace.area
        ownerElement::Cell = mesh.cells[theFace.iOwner]
        neighborElement::Cell = mesh.cells[theFace.iNeighbor]

        # vector between cell centroids
        CN::MVector{3, P} = neighborElement.centroid .- ownerElement.centroid
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
    for iBFace in (mesh.numInteriorFaces+1):length(mesh.faces)
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
    #     for i in 1:length(iNeighbors)
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

function labelBoundaryFaces(mesh::Mesh{P})::Mesh{P} where {P<:AbstractFloat}
    for (iBoundary, boundary) in enumerate(mesh.boundaries)
        startFace = boundary.startFace + 1
        nBFaces = boundary.nFaces
        for iFace in startFace:(startFace+nBFaces-1)
            mesh.faces[iFace].patchIndex = iBoundary
        end
    end
    return mesh
end # function labelBoundaryFaces

function readFields(dir::String, mesh::Mesh{P})::Tuple{Vector{Vector{BoundaryField}},Dict{Int32,String}} where {P<:AbstractFloat}
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
function parseVec(P::Type{<:AbstractFloat}, string::String)::MVector{3,P}
    cp = replace(string, "(" => "")
    cp = replace(cp, ")" => "")
    cp = replace(cp, ";" => "")
    cp = strip(cp)
    return parse.(P, split(cp, " "))
end

function parseVec(P::Type{<:AbstractFloat}, sub::SubString)::MVector{3,P}
    string = String(sub)
    cp = replace(string, "(" => "")
    cp = replace(cp, ")" => "")
    cp = replace(cp, ";" => "")
    cp = strip(cp)
    return parse.(P, split(cp, " "))
end
function isUniforminternalField(file::String)::Bool
    for line in eachline(file)
        if startswith(line, "internalField")
            return String(match(r"^(\w+)\s+(\w+)", line)[2]) == "uniform"
        end
    end
end

function getFieldType(P::Type{<:AbstractFloat}, file::String)
    for line in eachline(file)
        if contains(line, "class")
            if contains(line, "volVectorField")
                return MVector{3,P}
            end
            return P
        end
    end
end
"""
returns either P or MVector{3, P}
"""
function getUniformValue(P::Type{<:AbstractFloat}, file::String)
    for line in eachline(file)
        if startswith(line, "internalField")
            cpy = replace(line, r"internalField\s+uniform " => "")
            cpy = replace(cpy, "(" => "")
            cpy = replace(cpy, ")" => "")
            cpy = replace(cpy, ";" => "")
            values = parse.(P, split(cpy, " "))
            if length(values) == 1
                return values[1]
            end
            return MVector(values[1], values[2], values[3])
        end
    end
end

function readField(P::Type{<:AbstractFloat}, filePath::String, mesh::Mesh)::Tuple{Vector{BoundaryField{P}},Vector{SVector{3,P}}}
    fileName = rsplit(filePath, "/", limit=1)[1]
    if !isfile(filePath)
        throw(CaseDirError("Field file '$(filePath)' does not exist."))
    end
    isUniform = isUniforminternalField(filePath)
    # TODO so far not used
    numCells = length(mesh.cells)
    joined = ""
    fieldType = SVector{3,P} # getFieldType(filePath)
    internalValues = Vector{fieldType}(undef, numCells)
    if isUniform
        # println("isuniform")
        value = getUniformValue(P,filePath)
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
        if fieldType == SVector{3,P}
            for (j, line) in enumerate(eachline(filePath))
                if j < start
                    continue
                end
                if line == ")"
                    doBoundaries = true
                end
                if !doBoundaries
                    internalValues[index] = parse.(P, split(line[2:end-1], " "))
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
                    internalValues[index] = parse(P, line)
                else
                    joined = string(joined, line)
                end
            end
        end
    end
    splitted = split(joined, "}")
    boundaryFields = []
    for boundary in mesh.boundaries
        for b in splitted

            matches = match(r"\s*(\w+)\s*\{\s*type\s*(\w+);\s*(?:\s*value\s*(\w+)\s*(?:(d+)|(\([\d\.\s]+\)));)?", b)
            if isnothing(matches)
                continue
            end
            values::Vector{SVector{3,P}} = []
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
                    scalar = tryparse(P, matches[4])
                    values = fill(scalar, nFaces)
                else  # vectors like velocity
                    vector = parseVec(P, matches[5])
                    values = [SVector{3,P}(vector) for _ in 1:nFaces]
                end
            end
            bfield = BoundaryField(String(fileName), nFaces, values, String(type))
            push!(boundaryFields, bfield)
        end
    end
    return boundaryFields, internalValues
end

function readPropertiesFile(P::Type{<:AbstractFloat}, path::String)::P
    # println("Reading $(path)")
    if !isfile(path)
        throw(CaseDirError("Field file '$(path)' does not exist."))
    end
    file = read(path, String)
    variable = match(r"nu\s*(?:\[[\s\d-]+\])?\s*([\.\d\-e]+(?:E-\d)?);", file)
    val = tryparse(P, variable[1])
    return val
end

function upwind2(ϕf)
    if (ϕf >= 0)
        return 1.0
    end
    return 0.0
end

function upwind(ϕf::P)::P where {P<:AbstractFloat}
    # ϕf Uf ⋅ Sf = 0
    # ϕf is ̇m in the non-versteeg book
    if (ϕf >= 0)
        return one(P)
    end
    return zero(P)
end
struct s_upwind{T<:AbstractFloat} end
(u::s_upwind)(ϕf) = ϕf >= 0.0 ? 1.0 : 0.0

function centralDifferencing(_)::AbstractFloat
    return 0.5
end

function precalcWeights!(P::Type{<:AbstractFloat}, input::MatrixAssemblyInput)
    mesh = input.mesh
    U = input.U_internal
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
        input.weightsCdf[iFace] = centralDifferencing(ϕf)
        input.weightsUpwind[iFace] = upwind(ϕf)
    end
end

function prepareRelativeIndices!(input::MatrixAssemblyInput)
    mesh = input.mesh
    cells = mesh.cells
    for cell in cells
        ownerIdx = -1
        for iFace in 1:cell.nInternalFaces
            iFaceIndex = cell.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if ownerIdx == -1 && theFace.iOwner == cell.index && theFace.iOwner < theFace.iNeighbor
                theFace.relativeToOwner = 0
                ownerIdx = iFace
                continue
            end
            if cell.index == theFace.iOwner
                theFace.relativeToOwner = iFace - ownerIdx + 1
            else
                theFace.relativeToNeighbor = iFace - ownerIdx
            end
        end
        if ownerIdx == -1
            ownerIdx = cell.nInternalFaces + 1
        end
        for iFace in 1:cell.nInternalFaces
            iFaceIndex = cell.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if cell.index == theFace.iOwner
                theFace.relativeToOwner = iFace - ownerIdx + 1
            else
                theFace.relativeToNeighbor = iFace - ownerIdx
            end
        end
    end
end

function getOffsetsAndValues!(input::MatrixAssemblyInput)
    mesh = input.mesh
    nCells = length(mesh.cells)

    for iElement in 2:nCells
        input.offsets[iElement] += input.offsets[iElement-1] + mesh.cells[iElement-1].nInternalFaces
    end
    for iElement in 1:nCells
        theElement = mesh.cells[iElement]
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor > iElement
                input.negOffsets[theFace.iNeighbor] += 1
            end
        end
        input.offsets[iElement] += input.negOffsets[iElement]  # increase offset
    end
end

function ProcessCase(T::Type{<:AbstractFloat}, caseDirectory::String)::MatrixAssemblyInput
    mesh = readOpenFoamMesh(T, caseDirectory)
    # Define the thermal conductivity and source term
    nu = readPropertiesFile(T, joinpath(caseDirectory, "constant/transportProperties"))
    # Read initial condition and boundary conditions
    # p = readField(joinpath(caseDirectory, "0/p"), mesh)
    u_b, u_i = readField(T, joinpath(caseDirectory, "0/U"), mesh)
    # Assemble the coefficient matrix and RHS vector
    i = MatrixAssemblyInput(
        mesh,
        fill(nu, length(mesh.cells)),
        u_b,
        u_i,
        zeros(T, mesh.numInteriorFaces),
        zeros(T, mesh.numInteriorFaces),
        ones(Int32, length(mesh.cells)),
        zeros(Int32, length(mesh.cells))
    )
    precalcWeights!(T, i)
    prepareRelativeIndices!(i)
    getOffsetsAndValues!(i)
    return i
end # function ProcessCase
