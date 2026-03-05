include("classes.jl")
using LinearAlgebra
using StaticArrays

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
            0,
            -1
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
    return Mesh(nodes, faces, boundaries, numCells, cells, numInteriors, numBoundaryCells, numBoundaryFaces)
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
        if length(face.iNodes) == 3
            # sum x,y,z and divide by 3
            triangleNodes = [mesh.nodes[i] for i in face.iNodes]
            face.centroid .= sum(triangleNodes) / 3
            face.Sf .= 0.5 * cross(triangleNodes[2] - triangleNodes[1], triangleNodes[3] - triangleNodes[1])
            area = magnitude(face.Sf)
        else # general case, polygon is not a triangle
            nodes = [mesh.nodes[i] for i in face.iNodes]
            center = sum(nodes) / length(nodes)
            # Using the center to compute the area and centroid of virtual
            # triangles based on the center and the face nodes
            triangleNode1 = center
            triangleNode3 = MVector{3,Float32}(0.0, 0.0, 0.0)
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
    for iElement in 1:(length(mesh.cells))
        iFaces = mesh.cells[iElement].iFaces
        elementCenter = MVector{3,Float32}(0.0, 0.0, 0.0)
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

function processSecondaryFaceGeometry(mesh::Mesh)::Mesh
    # Loop over Interior faces
    eCN = MVector{3,Float32}(0.0, 0.0, 0.0)
    CN = MVector{3,Float32}(0.0, 0.0, 0.0)

    for iFace in 1:mesh.numInteriorFaces
        theFace::Face = mesh.faces[iFace]
        # Compute unit surtheFace normal vector
        # nf = theFace.Sf ./ theFace.area
        ownerElement::Cell = mesh.cells[theFace.iOwner]
        neighborElement::Cell = mesh.cells[theFace.iNeighbor]

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
            cpy = replace(line, r"internalField\s+uniform " => "")
            cpy = replace(cpy, "(" => "")
            cpy = replace(cpy, ")" => "")
            cpy = replace(cpy, ";" => "")
            values = parse.(Float32, split(cpy, " "))
            if length(values) == 1
                return values[1]
            end
            return MVector(values[1], values[2], values[3])
        end
    end
end

function readField(filePath::String, mesh::Mesh)
    fileName = rsplit(filePath, "/", limit=1)[1]
    if !isfile(filePath)
        throw(CaseDirError("Field file '$(filePath)' does not exist."))
    end
    isUniform = isUniforminternalField(filePath)
    # TODO so far not used
    numCells = length(mesh.cells)
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

            matches = match(r"\s*(\w+)\s*\{\s*type\s*(\w+);\s*(?:\s*value\s*(\w+)\s*(?:(d+)|(\([\d\.\s]+\)));)?", b)
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
    variable = match(r"nu\s*(?:\[[\s\d-]+\])?\s*([\.\d\-e]+(?:E-\d)?);", file)
    val = tryparse(Float32, variable[1])
    return val
end

function upwind(ϕf)
    # ϕf Uf ⋅ Sf = 0
    # ϕf is ̇m in the non-versteeg book
    if (ϕf >= 0)
        return 1.0
    end
    return 0.0
end

function centralDifferencing(_)
    return 0.5
end

function precalcWeights!(input::MatrixAssemblyInput)
    mesh = input.mesh
    U = input.U[2].values
    @inbounds for iFace in 1:mesh.numInteriorFaces
        theFace = mesh.faces[iFace]
        iOwner = theFace.iOwner
        iNeighbor = theFace.iNeighbor
        U_P = U[iOwner]
        U_N = U[iNeighbor]
        Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
        ϕf::Float32 = Uf ⋅ theFace.Sf                   # flux through the face
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

function ProcessCase(caseDirectory::String)::MatrixAssemblyInput
    mesh = readOpenFoamMesh(caseDirectory)
    # Define the thermal conductivity and source term
    nu = readPropertiesFile(joinpath(caseDirectory, "constant/transportProperties"))
    # Read initial condition and boundary conditions
    # p = readField(joinpath(caseDirectory, "0/p"), mesh)
    U = readField(joinpath(caseDirectory, "0/U"), mesh)
    # Assemble the coefficient matrix and RHS vector
    i = MatrixAssemblyInput(
        mesh,
        fill(nu, length(mesh.cells)),
        U,
        zeros(Float32, mesh.numInteriorFaces),
        zeros(Float32, mesh.numInteriorFaces),
        ones(Int32, length(mesh.cells)),
        zeros(Int32, length(mesh.cells))
    )
    precalcWeights!(i)
    prepareRelativeIndices!(i)
    getOffsetsAndValues!(i)
    return i
end # function ProcessCase
