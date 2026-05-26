include("common.jl")
using StaticArrays
using LinearAlgebra
using Atomix
function FusedCell(args, pde)
    backend = CUDABackend()
    FusedCellBasedKernel(backend, 64)(args..., pde; ndrange=length(args[6]))
    KernelAbstractions.synchronize(backend)
    return args[end-1:end]
end

@kernel function FusedCellBasedKernel(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(iFaceOffsets),
    @Const(facesPerCell),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(rowOffsets),
    @Const(ownerRelOwnerIdx),
    @Const(neighborRelNeighborIdx),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(iOwners),
    vals,
    RHS,
    @Const(fused_pde)
)
    nItems = @ndrange()
    iElement = @index(Global)
    t = eltype(nus)
    nCells = length(nus)
    numInternalFaces = length(Sf) - length(bFaceMapping)
    if iElement <= nCells
        diag, rhsx, rhsy, rhsz = zero(t), zero(t), zero(t), zero(t)
        @inbounds startIndex = iFaceOffsets[iElement] - Int32(1)
        @inbounds for iFace in one(Int32):numInteriors[iElement]
            @inbounds iFaceIndex = iFaces[startIndex+iFace]
            @inbounds valueUpper, valueLower = fused_pde(U[iElement], U[iNeighbors[iFaceIndex]], Sf[iFaceIndex], nus[iElement], gDiffs[iFaceIndex], 0.0, 0.0)
            if iElement == 1
                    @print("valueUpper, valueLower =  $valueUpper, $valueLower\n")
                end
            if iElement == iOwners[iFaceIndex]
                idx = ownerRelOwnerIdx[iFaceIndex]
                if iElement == 1
                    @print("idx: $idx\n")
                end
                Atomix.@atomic vals[idx] += valueLower
                diag += valueLower
            else
                idx = neighborRelNeighborIdx[iFaceIndex]
                if iElement == 1
                    @print("idx: $idx\n")
                end
                Atomix.@atomic vals[idx] += valueUpper
                diag += valueUpper
            end
        end
        # @inbounds for iFace in numInteriors[iElement]+1:facesPerCell[iElement]
        #     @inbounds iFaceIndex = iFaces[startIndex+iFace]
        #     @inbounds bFaceIndex = bFaceMapping[iFaceIndex-numInternalFaces]
        #     if bFaceIndex != -1
        #         @inbounds diag, rhsx, rhsy, rhsz = fused_pde(bFaceValues[bFaceIndex], Sf[iFaceIndex], nus[iElement], gDiffs[iFaceIndex], diag, rhsx, rhsy, rhsz)
        #     end
        # end
        # @inbounds RHS[iElement] += rhsx
        # @inbounds RHS[iElement+nCells] += rhsy
        # @inbounds RHS[iElement+nCells+nCells] += rhsz
        # @inbounds vals[rowOffsets[iElement]] += diag
    end
end

function PrecalculatedWeightsCell(args)
    backend = CUDABackend()
    PrecalculatedWeightsCellBasedKernel(backend, 64)(args...; ndrange=length(args[6]))
    KernelAbstractions.synchronize(backend)
    return args[end-1:end]
end

@kernel function PrecalculatedWeightsCellBasedKernel(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(iFaceOffsets),
    @Const(facesPerCell),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(rowOffsets),
    @Const(ownerRelOwnerIdx),
    @Const(neighborRelNeighborIdx),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(iOwners),
    vals,
    RHS,
    @Const(weights)
)
    iElement = @index(Global)
    t = eltype(nus)
    nCells = length(nus)
    numInternalFaces = length(Sf) - length(bFaceMapping)
    if iElement <= nCells
        diag, rhsx, rhsy, rhsz = zero(t), zero(t), zero(t), zero(t)
        @inbounds startIndex = iFaceOffsets[iElement] - Int32(1)
        @inbounds for iFace in one(Int32):numInteriors[iElement]
            @inbounds iFaceIndex = iFaces[startIndex+iFace]
            @inbounds isOwner = iOwners[iFaceIndex] == iElement

            flux = 0.5(U[iElement] + U[iNeighbors[iFace]]) ⋅ Sf[iFaceIndex]
            weights_f = weights[iFaceIndex]                      # get precalculated weight
            diffusion = nus[iElement] * gDiffs[iFaceIndex]          # laplacian(Γ, U)  ⟹ Diffusion
            valueUpper = flux * weights_f -diffusion
            valueLower = -flux * (1 - weights_f) + diffusion

            @inbounds idx = ifelse(isOwner, ownerRelOwnerIdx[iFaceIndex], neighborRelNeighborIdx[iFaceIndex])
            v = ifelse(isOwner, valueLower, valueUpper)
            Atomix.@atomic vals[idx] += v
            @inbounds diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        @inbounds for iFace in numInteriors[iElement]+1:facesPerCell[iElement]
            @inbounds iFaceIndex = iFaces[startIndex+iFace]
            @inbounds bFaceIndex = bFaceMapping[iFaceIndex-numInternalFaces]
            if bFaceIndex != -1
                convection = bFaceValues[bFaceIndex] .* Sf[iFace] ⋅ bFaceValues[bFaceIndex]
                diffusion = nus[iElement] * gDiffs[iFace]
                diag -= diffusion
                rhsx -= convection - diffusion
                rhsy -= convection - diffusion
                rhsz -= convection - diffusion    
            end          
        end
        @inbounds RHS[iElement] += rhsx
        @inbounds RHS[iElement+nCells] += rhsy
        @inbounds RHS[iElement+nCells+nCells] += rhsz
        Atomix.@atomic  vals[rowOffsets[iElement]] += diag
    end
end
function DynamicCell(args, fun)
    backend = CUDABackend()
    DynamicCellBasedKernel(backend, 64)(args..., fun; ndrange=length(args[6]))
    KernelAbstractions.synchronize(backend)
    return args[end-1:end]
end
@kernel function DynamicCellBasedKernel(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(iFaceOffsets),
    @Const(facesPerCell),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(rowOffsets),
    @Const(ownerRelOwnerIdx),
    @Const(neighborRelNeighborIdx),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(iOwners),
    vals,
    RHS,
    divScheme
)
    iElement = @index(Global)
    if iElement == 1 
        @print("type: $eltype(vals)")
    end
    # t = eltype(nus)
    # nCells = length(nus)
    # numInternalFaces = length(Sf) - length(bFaceMapping)
    # if iElement <= nCells
    #     diag, rhsx, rhsy, rhsz = zero(t), zero(t), zero(t), zero(t)
    #     @inbounds startIndex = iFaceOffsets[iElement] - Int32(1)
    #     @inbounds for iFace in one(Int32):numInteriors[iElement]
    #         @inbounds iFaceIndex = iFaces[startIndex+iFace]
    #         @inbounds isOwner = iOwners[iFaceIndex] == iElement

    #         flux = 0.5(U[iElement] + U[iNeighbors[iFace]]) ⋅ Sf[iFaceIndex]
    #         weights_f = divScheme(flux)                      # get precalculated weight
    #         diffusion = nus[iElement] * gDiffs[iFaceIndex]          # laplacian(Γ, U)  ⟹ Diffusion
    #         valueUpper = flux * weights_f -diffusion
    #         valueLower = -flux * (1 - weights_f) + diffusion

    #         @inbounds idx = ifelse(isOwner, ownerRelOwnerIdx[iFaceIndex], neighborRelNeighborIdx[iFaceIndex])
    #         v = ifelse(isOwner, valueLower, valueUpper)
    #         Atomix.@atomic vals[idx] = 0.0
    #         @inbounds diag += ifelse(!isOwner, valueLower, valueUpper)
    #     end
    #     @inbounds for iFace in numInteriors[iElement]+1:facesPerCell[iElement]
    #         @inbounds iFaceIndex = iFaces[startIndex+iFace]
    #         @inbounds bFaceIndex = bFaceMapping[iFaceIndex-numInternalFaces]
    #         if bFaceIndex != -1
    #             convection = bFaceValues[bFaceIndex] .* Sf[iFace] ⋅ bFaceValues[bFaceIndex]
    #             diffusion = nus[iElement] * gDiffs[iFace]
    #             diag -= diffusion
    #             rhsx -= convection - diffusion
    #             rhsy -= convection - diffusion
    #             rhsz -= convection - diffusion    
    #         end          
    #     end
    #     @inbounds RHS[iElement] += rhsx
    #     @inbounds RHS[iElement+nCells] += rhsy
    #     @inbounds RHS[iElement+nCells+nCells] += rhsz
    #     Atomix.@atomic  vals[rowOffsets[iElement]] += diag
    # end
end

function HardcodedCell(args, whichone)
    backend = CUDABackend()
    if whichone == "CDF"
        HardcodedCDFCellBasedKernel(backend, 64)(args...; ndrange=length(args[6]))
    else
        HardcodedUpwindCellBasedKernel(backend, 64)(args...; ndrange=length(args[6]))
    end
    KernelAbstractions.synchronize(backend)
    return args[end-1:end]
end

@kernel function HardcodedUpwindCellBasedKernel(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(iFaceOffsets),
    @Const(facesPerCell),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(rowOffsets),
    @Const(ownerRelOwnerIdx),
    @Const(neighborRelNeighborIdx),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(iOwners),
    vals,
    RHS
)
    iElement = @index(Global)
    t = eltype(nus)
    nCells = length(nus)
    numInternalFaces = length(Sf) - length(bFaceMapping)
    if iElement <= nCells
        diag, rhsx, rhsy, rhsz = zero(t), zero(t), zero(t), zero(t)
        @inbounds startIndex = iFaceOffsets[iElement] - Int32(1)
        @inbounds for iFace in one(Int32):numInteriors[iElement]
            @inbounds iFaceIndex = iFaces[startIndex+iFace]
            @inbounds isOwner = iOwners[iFaceIndex] == iElement

            flux = 0.5(U[iElement] + U[iNeighbors[iFace]]) ⋅ Sf[iFaceIndex]
            weights_f = upwind_f(flux)                      # get precalculated weight
            diffusion = nus[iElement] * gDiffs[iFaceIndex]          # laplacian(Γ, U)  ⟹ Diffusion
            valueUpper = flux * weights_f -diffusion
            valueLower = -flux * (1 - weights_f) + diffusion

            @inbounds idx = ifelse(isOwner, ownerRelOwnerIdx[iFaceIndex], neighborRelNeighborIdx[iFaceIndex])
            v = ifelse(isOwner, valueLower, valueUpper)
            Atomix.@atomic vals[idx] += v
            @inbounds diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        @inbounds for iFace in numInteriors[iElement]+1:facesPerCell[iElement]
            @inbounds iFaceIndex = iFaces[startIndex+iFace]
            @inbounds bFaceIndex = bFaceMapping[iFaceIndex-numInternalFaces]
            if bFaceIndex != -1
                convection = bFaceValues[bFaceIndex] .* Sf[iFace] ⋅ bFaceValues[bFaceIndex]
                diffusion = nus[iElement] * gDiffs[iFace]
                diag -= diffusion
                rhsx -= convection - diffusion
                rhsy -= convection - diffusion
                rhsz -= convection - diffusion    
            end          
        end
        @inbounds RHS[iElement] += rhsx
        @inbounds RHS[iElement+nCells] += rhsy
        @inbounds RHS[iElement+nCells+nCells] += rhsz
        Atomix.@atomic  vals[rowOffsets[iElement]] += diag
    end
end

@kernel function HardcodedCDFCellBasedKernel(
    @Const(iFaces),
    @Const(iNeighbors),
    @Const(numInteriors),
    @Const(iFaceOffsets),
    @Const(facesPerCell),
    @Const(nus),
    @Const(Sf),
    @Const(gDiffs),
    @Const(U),
    @Const(rowOffsets),
    @Const(ownerRelOwnerIdx),
    @Const(neighborRelNeighborIdx),
    @Const(bFaceValues),
    @Const(bFaceMapping),
    @Const(iOwners),
    vals,
    RHS
)
    iElement = @index(Global)
    t = eltype(nus)
    nCells = length(nus)
    numInternalFaces = length(Sf) - length(bFaceMapping)
    if iElement <= nCells
        diag, rhsx, rhsy, rhsz = zero(t), zero(t), zero(t), zero(t)
        @inbounds startIndex = iFaceOffsets[iElement] - Int32(1)
        @inbounds for iFace in one(Int32):numInteriors[iElement]
            @inbounds iFaceIndex = iFaces[startIndex+iFace]
            @inbounds isOwner = iOwners[iFaceIndex] == iElement

            flux = 0.5(U[iElement][1] + U[iNeighbors[iFaceIndex]][1]) *Sf[iFaceIndex][1] + 
                    0.5(U[iElement][2] + U[iNeighbors[iFaceIndex]][2]) *Sf[iFaceIndex][2] +
                    0.5(U[iElement][3] + U[iNeighbors[iFaceIndex]][3]) *Sf[iFaceIndex][3] 
            weights_f = cdf_f(flux)                      # get precalculated weight
            diffusion = nus[iElement] * gDiffs[iFaceIndex]          # laplacian(Γ, U)  ⟹ Diffusion
            valueUpper = flux * weights_f -diffusion
            valueLower = -flux * (1 - weights_f) + diffusion

            # # idx = ifelse(isOwner, ownerRelOwnerIdx[iFaceIndex], neighborRelNeighborIdx[iFaceIndex])
            # idx = isOwner ? ownerRelOwnerIdx[iFaceIndex] : neighborRelNeighborIdx[iFaceIndex]
            # # v = ifelse(isOwner, valueLower, valueUpper)
            # v = isOwner ? valueLower : valueUpper
            # # Atomix.@atomic 
            # vals[idx] += 0.0
            # # @inbounds diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        @inbounds for iFace in numInteriors[iElement]+1:facesPerCell[iElement]
            @inbounds iFaceIndex = iFaces[startIndex+iFace]
            @inbounds bFaceIndex = bFaceMapping[iFaceIndex-numInternalFaces]
            if bFaceIndex != -1
                flux = (Sf[iFaceIndex][1]) *bFaceValues[iFaceIndex][1] + 
                    (Sf[iFaceIndex][2]) *bFaceValues[iFaceIndex][2] +
                    (Sf[iFaceIndex][3]) *bFaceValues[iFaceIndex][3] 
                convection = bFaceValues[bFaceIndex] .* flux
                diffusion = nus[iElement] * gDiffs[iFace]
                diag -= diffusion
                rhsx -= convection - diffusion
                rhsy -= convection - diffusion
                rhsz -= convection - diffusion    
            end          
        end
        @inbounds RHS[iElement] += rhsx
        @inbounds RHS[iElement+nCells] += rhsy
        @inbounds RHS[iElement+nCells+nCells] += rhsz
        Atomix.@atomic  vals[rowOffsets[iElement]] += diag
    end
end

