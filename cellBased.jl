include("init.jl")
include("cpu_helper.jl")
function CellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets

    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0.0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = upwind(ϕf)                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)
            diffusion = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

            rel = theFace.relativeToOwner
            val = valueLower - diffusion
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper + diffusion
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper + diffusion
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            diffusion = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion
            diag -= diffusion
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = U_b .* ϕf
            value = convection .+ diffusion
            RHS[iElement] -= value[1]
            RHS[iElement+nCells] -= value[2]
            RHS[iElement+nCells+nCells] -= value[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end # function CellBasedAssembly

function LaplaceOnlyCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}) where {P<:AbstractFloat}
    mesh = input.mesh
    velocity_boundary = input.U_boundary
    nu = input.nu
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            diffusion = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

            rel = theFace.relativeToOwner
            val = -diffusion
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = diffusion
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += diffusion
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            @inbounds RHS[theFace.iOwner] -= diffusion[1]
            @inbounds RHS[theFace.iOwner+nCells] -= diffusion[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= diffusion[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function PrecalculatedWeightsUpwindCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}) where {P<:AbstractFloat}
    mesh = input.mesh
    weights = input.weightsUpwind
    velocity_boundary = input.U_boundary
    nu = input.nu
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = weights[theFace.index]                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)
            diffusion = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

            rel = theFace.relativeToOwner
            val = valueLower - diffusion
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper + diffusion
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper + diffusion
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function PrecalculatedWeightsCDFCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}) where {P<:AbstractFloat}
    mesh = input.mesh
    weights = input.weightsCdf
    nu = input.nu
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = weights[theFace.index]                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)
            diffusion = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

            rel = theFace.relativeToOwner
            val = valueLower - diffusion
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper + diffusion
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper + diffusion
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function DynamicCDFCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, div::Function) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)
            diffusion = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

            rel = theFace.relativeToOwner
            val = valueLower - diffusion
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper + diffusion
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper + diffusion
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function DynamicUpwindCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, div::Function) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)
            diffusion = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

            rel = theFace.relativeToOwner
            val = valueLower - diffusion
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper + diffusion
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper + diffusion
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function DivOnlyPrecalculatedWeightsUpwindCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}) where {P<:AbstractFloat}
    mesh = input.mesh
    weights = input.weightsUpwind
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = weights[theFace.index]                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)

            rel = theFace.relativeToOwner
            val = valueLower
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = U_b .* ϕf
            RHS[iElement] -= convection[1]
            RHS[iElement+nCells] -= convection[2]
            RHS[iElement+nCells+nCells] -= convection[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function DivOnlyPrecalculatedWeightsCDFCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}) where {P<:AbstractFloat}
    mesh = input.mesh
    weights = input.weightsCdf
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = weights[theFace.index]                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)

            rel = theFace.relativeToOwner
            val = valueLower
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = U_b .* ϕf
            RHS[iElement] -= convection[1]
            RHS[iElement+nCells] -= convection[2]
            RHS[iElement+nCells+nCells] -= convection[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function DivOnlyHardcodedUpwindCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}) where {P<:AbstractFloat}
    mesh = input.mesh
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = upwind(ϕf)                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)

            rel = theFace.relativeToOwner
            val = valueLower
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = U_b .* ϕf
            RHS[iElement] -= convection[1]
            RHS[iElement+nCells] -= convection[2]
            RHS[iElement+nCells+nCells] -= convection[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function DivOnlyHardcodedCDFCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}) where {P<:AbstractFloat}
    mesh = input.mesh
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = centralDifferencing(ϕf)                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)

            rel = theFace.relativeToOwner
            val = valueLower
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = U_b .* ϕf
            RHS[iElement] -= convection[1]
            RHS[iElement+nCells] -= convection[2]
            RHS[iElement+nCells+nCells] -= convection[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function DivOnlyDynamicCDFCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, div::Function) where {P<:AbstractFloat}
    mesh = input.mesh
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)

            rel = theFace.relativeToOwner
            val = valueLower
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = U_b .* ϕf
            RHS[iElement] -= convection[1]
            RHS[iElement+nCells] -= convection[2]
            RHS[iElement+nCells+nCells] -= convection[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function DivOnlyDynamicUpwindCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}, div::Function) where {P<:AbstractFloat}
    mesh = input.mesh
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)

            rel = theFace.relativeToOwner
            val = valueLower
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = U_b .* ϕf
            RHS[iElement] -= convection[1]
            RHS[iElement+nCells] -= convection[2]
            RHS[iElement+nCells+nCells] -= convection[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function HardcodedUpwindCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = upwind(ϕf)                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)
            diffusion = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

            rel = theFace.relativeToOwner
            val = valueLower - diffusion
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper + diffusion
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper + diffusion
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end

function HardcodedCDFCellBasedAssembly(input::MatrixAssemblyInput{P}, rows::Vector{Int32}, vals::Vector{P}, cols::Vector{Int32}, RHS::Vector{P}, offsets::Vector{Int32}) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U_boundary
    U = input.U_internal
    nCells = length(mesh.cells)
    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = centralDifferencing(ϕf)                      # get precalculated weight
            valueUpper = ϕf * weights_f
            valueLower = -ϕf * (1 - weights_f)
            diffusion = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

            rel = theFace.relativeToOwner
            val = valueLower - diffusion
            if theFace.iOwner != iElement
                rel = theFace.relativeToNeighbor
                val = valueUpper + diffusion
            end
            setIndex!(iElement, theElement.iNeighbors[iFace], val, rows, cols, vals, offsets[iElement] + rel)
            diag += valueUpper + diffusion
        end
        for iFace in theElement.nInternalFaces+1:numFaces
            iFaceIndex = theElement.iFaces[iFace]
            iBoundary = mesh.faces[iFaceIndex].patchIndex
            boundaryType = velocity_boundary[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            theFace = mesh.faces[iFaceIndex]
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            # convection
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
            ϕf = theFace.Sf ⋅ U_b
            @inbounds convection = velocity_boundary[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[theFace.iOwner] * theFace.gDiff
            setIndex!(theFace.iOwner, theFace.iOwner, -diffusion, rows, cols, vals, offsets[theFace.iOwner])
            # RHS/Source
            value = convection .+ diffusion
            @inbounds RHS[theFace.iOwner] -= value[1]
            @inbounds RHS[theFace.iOwner+nCells] -= value[2]
            @inbounds RHS[theFace.iOwner+nCells+nCells] -= value[3]
        end
        setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
    end
    return rows, cols, vals, RHS
end
