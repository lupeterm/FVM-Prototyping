include("init.jl")

function CellBasedAssembly2(input::MatrixAssemblyInput)
    println("Cellbased is currently broken.")
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    velocity_internal = input.U[2].values
    nCells = length(mesh.cells)
    RHS = zeros(Float32, nCells * 3)
    entriesNeeded = length(mesh.cells) + 2 * mesh.numInteriorFaces
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    vals = Vector{Float32}(undef, entriesNeeded * 3)
    idx = 1
    cellIdx = 1
    for iElement = 1:nCells
        set = false
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        @inbounds diagx::Float32 = velocity_internal[iElement][1]
        @inbounds diagy::Float32 = velocity_internal[iElement][2]
        @inbounds diagz::Float32 = velocity_internal[iElement][3]
        for iFace in 1:theElement.nInternalFaces
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
        for iFace in theElement.nInternalFaces+1:numFaces
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
end # function CellBasedAssembly



function ThreadedCellBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nCells = length(mesh.cells)
    RHS = zeros(Float32, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = getOffsets(input)
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

function CellBasedHelper(chunk::UnitRange, input::MatrixAssemblyInput, RHS::Vector{Float32}, rows::Vector{Int32}, cols::Vector{Int32}, offsets, vals::Vector)
    mesh = input.mesh
    nu = input.nu
    nCells = length(mesh.cells)
    velocity_boundary = input.U[1]
    velocity_internal = input.U[2].values
    for iElement in chunk
        theElement = mesh.cells[iElement]
        @inbounds diagx::Float32 = velocity_internal[iElement][1]
        @inbounds diagy::Float32 = velocity_internal[iElement][2]
        @inbounds diagz::Float32 = velocity_internal[iElement][3]
        @inbounds for iFace in 1:theElement.nInternalFaces
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
        for iFace in theElement.nInternalFaces+1:length(theElement.iFaces)
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
