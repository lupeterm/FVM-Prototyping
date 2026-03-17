include("init.jl")
using KernelAbstractions
function gpu_AllCellBasedAssemblyRunner()
end

struct GpuFace{P} where {P<:AbstractFloat}
    
end

@kernel function kernel_AllCellBasedAssemblyRunner(
    gpuCells,
    cellInternalFaces,
    gpuFaces,
    ops
)
    iElement = @index(Global)
    gpuCell = gpuCells[iElement]
    nInternals = cellInternalFaces[iElement]
    numFaces = length(theElement.iFaces)
    diag = 0.0
    for iFace in 1:nInternalFaces
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
        ϕf = theFace.Sf ⋅ U_b
        diag -= diffusion
        @inbounds convection = U_b .* ϕf
        value = convection .+ diffusion
        RHS[iElement] -= value[1]
        RHS[iElement+nCells] -= value[2]
        RHS[iElement+nCells+nCells] -= value[3]
    end
    setIndex!(iElement, iElement, diag, rows, cols, vals, offsets[iElement])
end