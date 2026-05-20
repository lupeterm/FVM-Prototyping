module MinimalFVM

include("operators.jl")
include("gpu.jl")
using PrecompileTools
using Atomix 

function cellBased_(
    temporals::Union{DiffEq,Ddt},
    volumes::Vector{P},
    oldVectors::Vector{P},
    oldOldVectors::Vector{P},
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    RHS::Vector{Float64}
) where {P<:AbstractFloat}
    nCells = Int32(length(oldVectors) / 3)
    for celli in 1:nCells
        idx2D = (celli - 1) * 3 + 1
        valueDiag, rx, ry, rz = temporals(
            oldVectors[idx2D:idx2D+2],
            oldOldVectors[idx2D:idx2D+2],
            volumes[celli],
            0.0,
            0.0,
            0.0,
            0.0
        )
        idx = rowOffs[celli] + 1 + diagOffs[celli]
        vals[idx] = valueDiag
        RHS[idx2D:idx2D+2] = [rx, ry, rz]
    end
end

function cellBased_(
    temporals::Union{DiffEq,Ddt},
    volumes::Vector{P},
    oldVectors::Vector{P},
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    RHS::Vector{Float64}
) where {P<:AbstractFloat}
    nCells = Int32(length(oldVectors) / 3)
    Threads.@threads for celli in 1:nCells
        idx2D = (celli - 1) * 3 + 1
        valueDiag, rx, ry, rz = temporals(
            oldVectors[idx2D:idx2D+2],
            volumes[celli],
            0.0,
            0.0,
            0.0,
            0.0
        )
        idx = rowOffs[celli] + 1 + diagOffs[celli]
        vals[idx] = valueDiag
        RHS[idx2D:idx2D+2] = [rx, ry, rz]
    end
end

function cellBased2(
    numCells::Int32,
    cellFacesSegments::Vector{Int32},
    diagOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    cellFacesValues::Vector{Int32},
    faceSignV::Vector{Float64},
    vals::Vector{Float64},
    opString::String,
    faceFlux::Vector{Float64},
    gamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    matrixColumnIdxV::Vector{Int32},
    volumes::Vector{Float64},
    oldVectors::Vector{Float64},
    RHS::Vector{Float64},
    dt::Float64
)
    dts = "$dt"
    opstring2 = replace(opString, "DELTAT" => dts)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")

    fused_pde = eval(Meta.parse(opstring3))
    ddt, spatials = MinimalFVM.splitTempSpat(fused_pde)
    temporals = !isnothing(ddt) ? ddt : f(args...) = (0.0, 0.0, 0.0, 0.0)
    for celli in 1:numCells
        diagValue = 0.0
        numFaces = cellFacesSegments[celli+1] - cellFacesSegments[celli]
        startIdx = cellFacesSegments[celli]
        for i in 1:numFaces
            faceIdx = cellFacesValues[startIdx+i] + 1
            sign = faceSignV[startIdx+i]

            offDiagValue, dVal = spatials(
                faceFlux[faceIdx],
                gamma[faceIdx],
                deltaCoeffs[faceIdx],
                magFaceArea[faceIdx],
                0.0,
                0.0
            )
            val[matrixColumnIdxV[startIdx+i]] += offDiagValue * sign
            diagValue -= dVal
        end
        idx2D = (celli - 1) * 3 + 1
        dv, rx, ry, rz = temporals(
            oldVectors[idx2D:idx2D+2],
            volumes[celli],
            0.0,
            0.0,
            0.0,
            0.0
        )
        diagValue += dv
        # Write diagonal and RHS
        diagIdx = rowOffs[celli] + 1 + diagOffs[celli]
        vals[diagIdx] += diagValue
        RHS[idx2D:idx2D+2] = [rx, ry, rz]
    end
end

# no temporals
function assemble(
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
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    fused_pde = eval(Meta.parse(opString))
    ddt, spatials = MinimalFVM.splitTempSpat(fused_pde)
    if !isnothing(ddt)
        println("[MinimalFVM] Tried assembly for $(typeof(ddt)), but no oldVectors were passed.")
    end
    if !isnothing(spatials)
        faceBasedAll_(numInteriorFaces,
            owner,
            neighbour,
            diagOffs,
            ownOffs,
            neiOffs,
            rowOffs,
            vals,
            spatials,
            faceFlux,
            gamma,
            deltaCoeffs,
            magFaceArea,
            valueFractions,
            refValue,
            refGradient,
            RHS,
            surfaceCells,
            bValues,
            bRhs
        )
    end
end

function assemble(
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
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64},
    volumes::Vector{Float64},
    oldVectors::Vector{Float64},
    dt::Float64
)
    # println("THIS IS CORRECT")
    dts = "$dt"
    opstring2 = replace(opString, "DELTAT" => dts)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")

    fused_pde = eval(Meta.parse(opstring3))
    ddt, spatials = MinimalFVM.splitTempSpat(fused_pde)
    if !isnothing(ddt)
        println("Calculating $ddt cell-based")
        cellBased_(
            ddt,
            volumes,
            oldVectors,
            diagOffs,
            rowOffs,
            vals,
            RHS
        )
    end
    if !isnothing(spatials)
        if Threads.nthreads() == 1
            println("Single threaded")
            faceBasedAll_(
                numInteriorFaces,
                owner,
                neighbour,
                diagOffs,
                ownOffs,
                neiOffs,
                rowOffs,
                vals,
                spatials,
                faceFlux,
                gamma,
                deltaCoeffs,
                magFaceArea,
                valueFractions,
                refValue,
                refGradient,
                RHS,
                surfaceCells,
                bValues,
                bRhs
            )
        else
            println("Multi threaded with $(Threads.nthreads())")
            faceBasedAll_threaded(
                numInteriorFaces,
                owner,
                neighbour,
                diagOffs,
                ownOffs,
                neiOffs,
                rowOffs,
                vals,
                spatials,
                faceFlux,
                gamma,
                deltaCoeffs,
                magFaceArea,
                valueFractions,
                refValue,
                refGradient,
                RHS,
                surfaceCells,
                bValues,
                bRhs
            )
        end
    end
end

function assemble(
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
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64},
    volumes::Vector{Float64},
    oldVectors::Vector{Float64},
    oldOldVectors::Vector{Float64},
    dt::Float64
)
    println("THIS IS WRONG")
    dts = "$dt"
    opstring = replace(opString, "DELTAT" => dts)
    println("after first replace: '$opstring'")
    fused_pde = eval(Meta.parse(opString))
    ddt, spatials = MinimalFVM.splitTempSpat(fused_pde)
    if !isnothing(ddt)
        cellBased_(
            temporals,
            volumes,
            oldVectors,
            oldOldVectors,
            diagOffs,
            rowOffs,
            vals,
            RHS
        )
    end
    if !isnothing(spatials)
        faceBasedAll_(numInteriorFaces,
            owner,
            neighbour,
            diagOffs,
            ownOffs,
            neiOffs,
            rowOffs,
            vals,
            spatials,
            faceFlux,
            gamma,
            deltaCoeffs,
            magFaceArea,
            valueFractions,
            refValue,
            refGradient,
            RHS,
            surfaceCells,
            bValues,
            bRhs
        )
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
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    fused_pde = eval(Meta.parse(opString))
    faceBasedAll_(
        numInteriorFaces,
        owner,
        neighbour,
        diagOffs,
        ownOffs,
        neiOffs,
        rowOffs,
        vals,
        fused_pde,
        faceFlux,
        gamma,
        deltaCoeffs,
        magFaceArea,
        valueFractions,
        refValue,
        refGradient,
        RHS,
        surfaceCells,
        bValues,
        bRhs
    )
end

function faceBasedAll_(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    fused_pde::PTERM,
    faceFlux::Vector{Float64},
    gamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    numCells = length(rowOffs) - 1
    for iFace in 1:numInteriorFaces
        iOwner = owner[iFace] + 1
        iNeighbor = neighbour[iFace] + 1

        rowNeiStart = rowOffs[iNeighbor]
        rowOwnStart = rowOffs[iOwner]

        valueUpper, valueLower = fused_pde(
            faceFlux[iFace], 
            gamma[iFace], 
            deltaCoeffs[iFace], 
            magFaceArea[iFace], 
            0.0, 
            0.0
        )
        idx = (rowNeiStart + neiOffs[iFace]) * 3 + 1
        vals[idx:idx+2] .+= valueUpper
        idx = (rowOwnStart + diagOffs[iOwner]) * 3 + 1
        vals[idx:idx+2] .-= valueUpper
        idx = (rowOwnStart + ownOffs[iFace]) * 3 + 1
        vals[idx:idx+2] .+= valueLower
        idx = (rowNeiStart + diagOffs[iNeighbor]) * 3 + 1
        vals[idx:idx+2] .-= valueLower
    end
    for facei in numInteriorFaces+1:length(faceFlux)
        bcfacei = facei - numInteriorFaces
        start = bcfacei * 3 - 2
        end_ = start + 2
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

        vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1
        vals[vIdx:vIdx+2] .+= valueDiag

        bValues[bcfacei*3-2:bcfacei*3] .= valueDiag

        # rhs[own] -= valueRhs
        # FIXME dont forget, changed this back to [vec3, vec3] instead of [xxxyyyzzz] for now
        RHS[own*3-2:own*3] += [valueRHSx, valueRHSy, valueRHSz]
        # RHS[own] += valueRHSx
        # RHS[own+numCells] += valueRHSy
        # RHS[own+numCells+numCells] += valueRHSz

        bRhs[bcfacei*3-2:bcfacei*3] += [valueRHSx, valueRHSy, valueRHSz]
        # bRhs[own] += valueRHSx
        # bRhs[own+numCells] += valueRHSy
        # bRhs[own+numCells+numCells] += valueRHSz
    end
end

function faceBasedAll_threaded(
    numInteriorFaces::Int32,
    owner::Vector{Int32},
    neighbour::Vector{Int32},
    diagOffs::Vector{UInt8},
    ownOffs::Vector{UInt8},
    neiOffs::Vector{UInt8},
    rowOffs::Vector{Int32},
    vals::Vector{Float64},
    fused_pde::PTERM,
    faceFlux::Vector{Float64},
    gamma::Vector{Float64},
    deltaCoeffs::Vector{Float64},
    magFaceArea::Vector{Float64},
    valueFractions::Vector{Float64},
    refValue::Vector{Float64},
    refGradient::Vector{Float64},
    RHS::Vector{Float64},
    surfaceCells::Vector{Int32},
    bValues::Vector{Float64},
    bRhs::Vector{Float64}
)
    numCells = length(rowOffs) - 1
    Threads.@threads for iFace in 1:numInteriorFaces
        iOwner = owner[iFace] + 1
        iNeighbor = neighbour[iFace] + 1

        rowNeiStart = rowOffs[iNeighbor]
        rowOwnStart = rowOffs[iOwner]

        valueUpper, valueLower = fused_pde(faceFlux[iFace], gamma[iFace], deltaCoeffs[iFace], magFaceArea[iFace], 0.0, 0.0)
        idx = (rowNeiStart + neiOffs[iFace]) * 3 + 1
        vals[idx:idx+2] .+= valueUpper
        idx = (rowOwnStart + diagOffs[iOwner]) * 3 + 1
        vals[idx:idx+2] .-= valueUpper
        idx = (rowOwnStart + ownOffs[iFace]) * 3 + 1
        vals[idx:idx+2] .+= valueLower
        idx = (rowNeiStart + diagOffs[iNeighbor]) * 3 + 1
        vals[idx:idx+2] .-= valueLower
    end
    Threads.@threads for facei in numInteriorFaces+1:length(faceFlux)
        bcfacei = facei - numInteriorFaces
        start = bcfacei * 3 - 2
        end_ = start + 2
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

        vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1
        if bcfacei == 1
            println("writing in $vIdx : $(vIdx+2)")
        end
        vals[vIdx:vIdx+2] .+= valueDiag

        bValues[bcfacei*3-2:bcfacei*3] .= valueDiag

        # rhs[own] -= valueRhs
        # FIXME dont forget, changed this back to [vec3, vec3] instead of [xxxyyyzzz] for now
        RHS[own*3-2:own*3] += [valueRHSx, valueRHSy, valueRHSz]
        # RHS[own] += valueRHSx
        # RHS[own+numCells] += valueRHSy
        # RHS[own+numCells+numCells] += valueRHSz

        bRhs[bcfacei*3-2:bcfacei*3] += [valueRHSx, valueRHSy, valueRHSz]
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
    assemble(
        Int32(1),
        ones(Int32, 2),
        ones(Int32, 2),
        ones(UInt8, 2),
        ones(UInt8, 2),
        ones(UInt8, 2),
        ones(Int32, 2),
        zeros(Float64, 12),
        op,
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 2),
        ones(Float64, 3),
        ones(Float64, 3),
        zeros(Float64, 12),
        zeros(Int32, 2),
        zeros(Float64, 12),
        zeros(Float64, 12),
        zeros(Float64, 2),
        zeros(Float64, 12),
        0.001,
    )
end


@compile_workload begin
    warmup("Div{Float64, linear{Float64}}(linear{Float64}(), 1)")
    warmup("Laplace{Float64}(1)")
    warmup("Div{Float64, upwind{Float64}}(upwind{Float64}(), 1) + Laplace{Float64}(-5)")
    warmup("Div{Float64, upwind{Float64}}(upwind{Float64}(), 1) + Laplace{Float64}(1)")
end
export faceBased, Div, Noop, linear, upwind, BDF1, Laplace, test, faceBasedBoundary, faceBasedAll, Ddt, hasTransient, splitTempSpat, BDF2, DELTAT, assemble_gpu

end # module MinimalFVM
