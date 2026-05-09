module MinimalFVM

include("operators.jl")
using PrecompileTools
using SnoopCompile, AbstractTrees, SnoopCompileCore

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


@compile_workload begin
    d = eval(Meta.parse("Div{Float64, upwind{Float64}}(upwind{Float64}(), 1.0)"))
    d = eval(Meta.parse("Laplace{Float64}(1.0)"))
    d = eval(Meta.parse("Div{Float64, upwind{Float64}}(upwind{Float64}(), 1.0) + Laplace{Float64}(1.0)"))
    laplace = Laplace{Float64}(1.0)
    laplace(1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    divU = Div{Float64,upwind{Float64}}(upwind{Float64}(), 1.0)
    divU(1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    divULap = divU + laplace
    divL = Div{Float64,linear{Float64}}(linear{Float64}(), 1.0)
    divL(1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    divLLap = divL + laplace
    divLLap(1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    a = ones(Float64, 3, 10)
    a2 = [Vector(a[:, i]) for i in axes(a, 2)]
    divU(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.0,
        1.0,
        6.0,
        3.0,
        0.01,
        0.1111111111111111,
        0.0,
        0.0,
        0.0,
        0.0
    )
    divL(
        ones(Float64, 3),
        ones(Float64, 3),
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0
    )
    laplace(
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0
    )
    faceBased(
        Int32(1),
        zeros(Int32, 1),
        zeros(Int32, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(Int32, 1),
        zeros(Float64, 1),
        "Div{Float64, upwind}(upwind{Float64}(), 1.0)",
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
    )
    faceBased(
        Int32(1),
        zeros(Int32, 1),
        zeros(Int32, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(Int32, 1),
        zeros(Float64, 1),
        "Div{Float64, linear}(linear{Float64}(), 1.0)",
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
    )
    faceBased(
        Int32(1),
        zeros(Int32, 1),
        zeros(Int32, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(Int32, 1),
        zeros(Float64, 1),
        "Laplace{Float64}(1.0)",
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
    )
    faceBased(
        Int32(1),
        zeros(Int32, 1),
        zeros(Int32, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(Int32, 1),
        zeros(Float64, 1),
        "Div{Float64, linear}(linear{Float64}(), 1.0) + Laplace{Float64}(1.0)",
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
    )
    faceBased(
        Int32(1),
        zeros(Int32, 1),
        zeros(Int32, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(UInt8, 1),
        zeros(Int32, 1),
        zeros(Float64, 1),
        "Div{Float64, upwind{Float64}}(upwind{Float64}(), 1.0) + Laplace{Float64}(1.0)",
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
        zeros(Float64, 1),
    )

    faceBasedBoundary(
        Int32(3),
        ones(Int32, 3),
        ones(UInt8, 3),
        ones(Int32, 3),
        zeros(Float64, 3),
        "Div{Float64, upwind{Float64}}(upwind{Float64}(), 1.0)",
        ones(Float64, 3),
        ones(Float64, 3),
        ones(Float64, 3),
        ones(Float64, 3),
        ones(Float64, 3),
        ones(Float64, 3, 12),
        ones(Float64, 3, 12),
        ones(Float64, 3),
    )
    faceBasedBoundary(
        Int32(3),
        ones(Int32, 3),
        ones(UInt8, 3),
        ones(Int32, 3),
        zeros(Float64, 3),
        "Laplace{Float64}(1.0)",
        ones(Float64, 3),
        zeros(Float64, 3),
        zeros(Float64, 3),
        zeros(Float64, 3),
        ones(Float64, 3),
        zeros(Float64, 3, 12),
        zeros(Float64, 3, 12),
        zeros(Float64, 3),
    )
    # numInteriorFaces::Int32,
    # owner::Vector{Int32},  # surfaceCells 
    # diagOffs::Vector{UInt8},
    # rowOffs::Vector{Int32},
    # vals::Vector{Float64},
    # opString::String,
    # faceFlux::Vector{Float64},
    # gamma::Vector{Float64},
    # deltaCoeffs::Vector{Float64},
    # magFaceArea::Vector{Float64},
    # valueFractions::Vector{Float64},
    # refValue_::Matrix{Float64},
    # refGradient_::Matrix{Float64},
    # RHS::Vector{Float64}
    faceBasedBoundary(
        Int32(3),
        ones(Int32, 3),
        ones(UInt8, 3),
        ones(Int32, 3),
        zeros(Float64, 3),
        "Div{Float64, upwind{Float64}}(upwind{Float64}(), 1.0) + Laplace{Float64}(1.0)",
        ones(Float64, 3),
        zeros(Float64, 3),
        zeros(Float64, 3),
        zeros(Float64, 3),
        zeros(Float64, 3),
        zeros(Float64, 3, 12),
        zeros(Float64, 3, 12),
        zeros(Float64, 3),
    )
    op = Div{Float64,upwind{Float64}}(upwind{Float64}(), 1.0)
    a = ones(Float64, 3)
    op(
        a, a,
        one(Float64),
        one(Float64),
        6.0,
        3.0,
        one(Float64),
        one(Float64),
        0.0, 0.0, 0.0, 0.0
    )
end
export faceBased, Div, Noop, linear, upwind, BDF1, Laplace, test, faceBasedBoundary

end # module MinimalFVM
