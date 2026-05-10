module MinimalFVM

include("operators.jl")
using PrecompileTools
using SnoopCompile, AbstractTrees, SnoopCompileCore

# function cellBased()
#     for celli in 1:numCells
#         diagIdx = rowOffs[celli] + diaOffV[celli];
#         #  const auto coeff = a0 * vol[celli];
#         #  auto diagValue = coeff * one<ValueType>();
#         #  auto rhsValue = coeff * oldVector[celli];

#         # Loop over faces of this cell
#         diagValue = zero(Float64);
#         numFaces = cellFacesSegments[celli + 1] - cellFacesSegments[celli];
#         startIdx = cellFacesSegments[celli];

#         for i in 1:numFaces
#         {
#             faceIdx = cellFacesValues[startIdx + i];
#             neiCell = faceNeighbourV[startIdx + i];
#             sign = faceSignV[startIdx + i];

#             # Compute flux on-the-fly
#             fluxDiv = phiV[faceIdx]; # faceFluxV[faceIdx];
#             weight = (phiV[faceIdx] >= 0) ? 0.0 : 1.0; // weightsV[faceIdx];
#             lapFlux = deltaV[faceIdx] * gammaV[faceIdx] * magFaceAreaV[faceIdx];
#             combinedFlux = (-weight * fluxDiv + lapFlux) * one<ValueType>();

#             offDiagValue = sign * combinedFlux;
#             matrix.values[matrixColumnIdxV[startIdx + i]] += offDiagValue;

#             # // Contribution to diagonal (subtract off-diagonal)
#             diagValue -= offDiagValue;
#         }

#         # // Write diagonal and RHS
#         matrix.values[diagIdx] += diagValue + a0a1 * one<ValueType>();
#         # // FIXME
#         # // const auto commonCoef = operatorScaling[celli] * vol[celli];
#         rhs[celli] += a0a1 * oldVector[celli];
#     end
# end

function test(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    opString::Cstring,
    faceFlux::Vector{Float64},
    gamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Matrix{Float64},
    refGradient_::Matrix{Float64},
    RHS::Vector{Float64}
)
    s = unsafe_string(opString)
    fused_pde = eval(Meta.parse(s))
    numCells = length(rowOffs) - 1
    for iFace in 1:numInteriorFaces
        iOwner = owner[iFace] + 1
        iNeighbor = neighbour[iFace] + 1

        rowNeiStart = rowOffs[iNeighbor] + 1
        rowOwnStart = rowOffs[iOwner] + 1

        valueUpper, valueLower = fused_pde(faceFlux[iFace], gamma[iFace], deltaCoeffs[iFace], magFaceArea[iFace], 0.0, 0.0)
        vals[rowNeiStart+neiOffs[iFace]] += valueUpper
        vals[rowOwnStart+diagOffs[iOwner]] -= valueUpper
        vals[rowOwnStart+ownOffs[iFace]] += valueLower
        vals[rowNeiStart+diagOffs[iNeighbor]] -= valueLower
    end
    for facei in numInteriorFaces+1:length(faceFlux)
        bcfacei = facei - numInteriorFaces
        valueDiag, valueRHSx, valueRHSy, valueRHSz = 0.0,0.0,0.0,0.0
        # = fused_pde(
        #     refValue[:, bcfacei],
        #     refGradient_[:, bcfacei],
        #     faceFlux[facei],
        #     valueFractions[bcfacei],
        #     6.0,
        #     3.0,
        #     gamma[facei],
        #     magFaceArea[facei],
        #     0.0, 0.0, 0.0, 0.0
        # )
        own = owner[bcfacei] + 1
        rowOwnStart = rowOffs[own] + 1
        # operatorScalingOwn = operatorScaling[own]

        # valueMat = flux * operatorScalingOwn * valFrac2

        vals[rowOwnStart+diagOffs[own]] += valueDiag
        # bValues[bcfacei] = valueDiag
        # rhs[own] -= valueRhs
        RHS[own] += valueRHSx
        RHS[own+numCells] += valueRHSy
        RHS[own+numCells+numCells] += valueRHSz

        # bRhs[own] += valueRHSx
        # bRhs[own+numCells] += valueRHSy
        # bRhs[own+numCells+numCells] += valueRHSz
    end
end

function faceBasedAll(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    gamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32}
)
    fused_pde = eval(Meta.parse(opString))
    numCells = length(rowOffs) - 1
    for iFace in 1:numInteriorFaces
        iOwner = owner[iFace] + 1
        iNeighbor = neighbour[iFace] + 1

        rowNeiStart = rowOffs[iNeighbor] + 1
        rowOwnStart = rowOffs[iOwner] + 1

        valueUpper, valueLower = fused_pde(faceFlux[iFace], gamma[iFace], deltaCoeffs[iFace], magFaceArea[iFace], 0.0, 0.0)
        vals[rowNeiStart+neiOffs[iFace]] += valueUpper
        vals[rowOwnStart+diagOffs[iOwner]] -= valueUpper
        vals[rowOwnStart+ownOffs[iFace]] += valueLower
        vals[rowNeiStart+diagOffs[iNeighbor]] -= valueLower
    end
    for facei in numInteriorFaces+1:length(faceFlux)
        bcfacei = facei - numInteriorFaces
        start = bcfacei*3-2 
        end_ = start+2  
        valueDiag, valueRHSx, valueRHSy, valueRHSz = fused_pde(
            refValue[start:end_],
            refGradient[start:end_],
            faceFlux[facei],
            valueFractions[bcfacei],
            6.0,
            3.0,
            gamma[facei],
            magFaceArea[facei],
            0.0, 0.0, 0.0, 0.0
        )
        own = surfaceCells[bcfacei] + 1
        rowOwnStart = rowOffs[own] + 1
        # operatorScalingOwn = operatorScaling[own]

        vals[rowOwnStart+diagOffs[own]] += valueDiag
        # bValues[bcfacei] = valueDiag
        # rhs[own] -= valueRhs
        RHS[own] += valueRHSx
        RHS[own+numCells] += valueRHSy
        RHS[own+numCells+numCells] += valueRHSz

        # bRhs[own] += valueRHSx
        # bRhs[own+numCells] += valueRHSy
        # bRhs[own+numCells+numCells] += valueRHSz
    end
end


function faceBased(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    gamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64}
)
    fused_pde = eval(Meta.parse(opString))
    for iFace in 1:numInteriorFaces
        iOwner = owner[iFace] + 1
        iNeighbor = neighbour[iFace] + 1

        rowNeiStart = rowOffs[iNeighbor] + 1
        rowOwnStart = rowOffs[iOwner] + 1

        valueUpper, valueLower = fused_pde(faceFlux[iFace], gamma[iFace], deltaCoeffs[iFace], magFaceArea[iFace], 0.0, 0.0)
        vals[rowNeiStart+neiOffs[iFace]] += valueUpper
        vals[rowOwnStart+diagOffs[iOwner]] -= valueUpper
        vals[rowOwnStart+ownOffs[iFace]] += valueLower
        vals[rowNeiStart+diagOffs[iNeighbor]] -= valueLower
    end
end

function faceBasedBoundary(
    numInteriorFaces::Int32,
    owner::Vector{Int32},  # surfaceCells 
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    gamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},  # FIXME
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Matrix{Float64},
    refGradient_::Matrix{Float64},
    RHS::Vector{Float64}
)

    fused_pde = eval(Meta.parse(opString))
    numCells = length(rowOffs) - 1
    for facei in numInteriorFaces+1:length(faceFlux)
        bcfacei = facei - numInteriorFaces
        valueDiag, valueRHSx, valueRHSy, valueRHSz = fused_pde(
            refValue[:, bcfacei],
            refGradient_[:, bcfacei],
            faceFlux[facei],
            valueFractions[bcfacei],
            6.0,
            3.0,
            gamma[facei],
            magFaceArea[facei],
            0.0, 0.0, 0.0, 0.0
        )
        @inbounds own = owner[bcfacei] + 1
        @inbounds rowOwnStart = rowOffs[own] + 1
        # operatorScalingOwn = operatorScaling[own]

        # valueMat = flux * operatorScalingOwn * valFrac2

        @inbounds vals[rowOwnStart+diagOffs[own]] += valueDiag
        # bValues[bcfacei] = valueDiag
        # rhs[own] -= valueRhs
        @inbounds RHS[own] += valueRHSx
        @inbounds RHS[own+numCells] += valueRHSy
        @inbounds RHS[own+numCells+numCells] += valueRHSz
        # bRhs[own] += valueRHSx
        # bRhs[own+numCells] += valueRHSy
        # bRhs[own+numCells+numCells] += valueRHSz
    end
end

function warmup(op::String)
    faceBasedAll(
        Int32(1),
        ones(Int32, 2),
        ones(Int32, 2),
        ones(UInt8, 2),
        ones(UInt8, 2),
        ones(UInt8, 2),
        ones(Int32, 2),
        zeros(Float64, 4),
        op,
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 3),
        ones(Float64, 3),
        zeros(Float64, 9),
        zeros(Int32, 2),
    )
end


@compile_workload begin
    warmup("Div{Float64, linear{Float64}}(linear{Float64}(), 1)")
    warmup("Laplace{Float64}(1)")
    warmup("Div{Float64, upwind{Float64}}(upwind{Float64}(), 1) + Laplace{Float64}(-5)")
    warmup("Div{Float64, upwind{Float64}}(upwind{Float64}(), 1) + Laplace{Float64}(1)")

end
export faceBased, Div, Noop, linear, upwind, BDF1, Laplace, test, faceBasedBoundary, faceBasedAll

end # module MinimalFVM
