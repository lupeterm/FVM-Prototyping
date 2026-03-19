include("init.jl")
include("cpu_helper.jl")
function CellBasedAssembly(input::MatrixAssemblyInput)
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
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = upwind(ϕf)                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
            diffusion::P = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

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

function LaplaceOnlyCellBasedAssembly(input::MatrixAssemblyInput)
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
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            diffusion::P = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

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
            relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
            U_b = velocity_boundary[iBoundary].values[relativeFaceIndex]
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

function PrecalculatedWeightsUpwindCellBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    weights = input.weightsUpwind
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets

    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = weights[theFace.index]                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
            diffusion::P = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

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

function PrecalculatedWeightsCDFCellBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    weights = input.weightsCdf
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets

    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = weights[theFace.index]                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
            diffusion::P = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

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

function DynamicCDFCellBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    weights = input.weightsCdf
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets

    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
            diffusion::P = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

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

function DynamicUpwindCellBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    weights = input.weightsCdf
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets

    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
            diffusion::P = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

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

function DivOnlyPrecalculatedWeightsUpwindCellBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    weights = input.weightsUpwind
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets

    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = weights[theFace.index]                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)

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

function DivOnlyPrecalculatedWeightsCDFCellBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nu = input.nu
    velocity_boundary = input.U[1]
    U = input.U[2].values
    nCells = length(mesh.cells)
    weights = input.weightsCdf
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    vals = zeros(P, entriesNeeded)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    RHS = zeros(P, nCells * 3)
    offsets = input.offsets

    for iElement::Int32 in eachindex(mesh.cells)
        theElement = mesh.cells[iElement]
        numFaces = length(theElement.iFaces)
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = weights[theFace.index]                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)

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

function DivOnlyHardcodedUpwindCellBasedAssembly(input::MatrixAssemblyInput)
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
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = upwind(ϕf)                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)

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

function DivOnlyHardcodedCDFCellBasedAssembly(input::MatrixAssemblyInput)
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
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = centralDifferencing(ϕf)                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)

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

function DivOnlyDynamicCDFCellBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
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
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)

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

function DivOnlyDynamicUpwindCellBasedAssembly(input::MatrixAssemblyInput, div::Function)
    mesh = input.mesh
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
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = div(ϕf)                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)

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

function HardcodedUpwindCellBasedAssembly(input::MatrixAssemblyInput)
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
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = upwind(ϕf)                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
            diffusion::P = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

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

function HardcodedCDFCellBasedAssembly(input::MatrixAssemblyInput)
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
        diag = 0
        for iFace in 1:theElement.nInternalFaces
            iFaceIndex = theElement.iFaces[iFace]
            theFace::Face = mesh.faces[iFaceIndex]
            U_P = U[iElement]
            U_N = U[theElement.iNeighbors[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf::P = Uf ⋅ theFace.Sf                   # flux through the face
            weights_f = centralDifferencing(ϕf)                      # get precalculated weight
            valueUpper::P = ϕf * weights_f
            valueLower::P = -ϕf * (1 - weights_f)
            diffusion::P = nu[iElement] * theFace.gDiff          # laplacian(Γ, U)  ⟹ Diffusion

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


### keep?
#    |
#    v

function ThreadedCellBasedAssembly(input::MatrixAssemblyInput)
    mesh = input.mesh
    nCells = length(mesh.cells)
    RHS = zeros(P, nCells * 3)
    entriesNeeded::Int32 = length(mesh.cells) + 2 * mesh.numInteriorFaces
    offsets = getOffsets(input)
    rows = zeros(Int32, entriesNeeded)
    cols = zeros(Int32, entriesNeeded)
    vals = Vector{P}(undef, entriesNeeded * 3)
    chunks = Iterators.partition(1:nCells, nCells ÷ nthreads())
    chunks = [c for c in chunks]
    @threads for chunk in chunks
        CellBasedHelper(chunk, input, RHS, rows, cols, offsets, vals)
    end
    return rows, cols, vals
end

function CellBasedHelper(chunk::UnitRange, input::MatrixAssemblyInput, RHS::Vector{P}, rows::Vector{Int32}, cols::Vector{Int32}, offsets, vals::Vector) where {P<:AbstractFloat}
    mesh = input.mesh
    nu = input.nu
    nCells = length(mesh.cells)
    velocity_boundary = input.U[1]
    velocity_internal = input.U[2].values
    for iElement in chunk
        theElement = mesh.cells[iElement]
        @inbounds diagx::P = velocity_internal[iElement][1]
        @inbounds diagy::P = velocity_internal[iElement][2]
        @inbounds diagz::P = velocity_internal[iElement][3]
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
                fluxVb::Vector{P} = velocity_boundary[iBoundary].values[relativeFaceIndex] .* -fluxCb
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
