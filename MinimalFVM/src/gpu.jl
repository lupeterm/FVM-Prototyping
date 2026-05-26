using CUDA
using KernelAbstractions
using Atomix
using StaticArrays

function ptrToGpu(ptr::Ptr{Cvoid}, size, dType)
	p = reinterpret(CuPtr{dType}, ptr)
    arr = unsafe_wrap(CuArray, p, Int64(size); own=false)
	return arr
end

function updateIDelta(value, size)
    Core.eval(@__MODULE__, Expr(:(=), :IDELTACOEFFS, ptrToGpu(value, size, Float64)))
end

function updateBDelta(value, size)
    Core.eval(@__MODULE__, Expr(:(=), :BDELTACOEFFS, ptrToGpu(value, size, Float64)))
end

function assemble_gpu(
    numInteriorFaces::Int32,
	_owner::Ptr{Cvoid},
	_neighbour::Ptr{Cvoid},
	_diagOffs::Ptr{Cvoid},
	_ownOffs::Ptr{Cvoid},
	_neiOffs::Ptr{Cvoid},
	_rowOffs::Ptr{Cvoid},
	_vals::Ptr{Cvoid},
	opString::String,
	_faceFlux::Ptr{Cvoid},
	_bfaceFlux::Ptr{Cvoid},
	_gamma::Ptr{Cvoid},
	_bgamma::Ptr{Cvoid},
	_deltaCoeffs::Ptr{Cvoid},
	_bdeltaCoeffs::Ptr{Cvoid},
	_magFaceArea::Ptr{Cvoid},
	_valueFractions::Ptr{Cvoid},
	_refValue::Ptr{Cvoid},
	_refGradient::Ptr{Cvoid},
	_RHS::Ptr{Cvoid},
	_surfaceCells::Ptr{Cvoid},
	_bValues::Ptr{Cvoid},
	_bRhs::Ptr{Cvoid},
	_volumes::Ptr{Cvoid},
	_oldVectors::Ptr{Cvoid},
	_dt::Float64,
	numCells::Int32,
	numTotalFaces::Int32
)
	boundaryFaces = numTotalFaces - numInteriorFaces
	owner = ptrToGpu(_owner, numInteriorFaces, Int32)	
	neighbour = ptrToGpu(_neighbour, numInteriorFaces, Int32)
	diagOffs = ptrToGpu(_diagOffs, numCells, UInt8)
	ownOffs = ptrToGpu(_ownOffs, numInteriorFaces, UInt8)
	neiOffs = ptrToGpu(_neiOffs, numInteriorFaces, UInt8)
	rowOffs = ptrToGpu(_rowOffs, numCells, Int32)
	vals = ptrToGpu(_vals, (numCells + 2*numInteriorFaces)*3, Float64)
	faceFlux = ptrToGpu(_faceFlux, numInteriorFaces, Float64)
	bfaceFlux = ptrToGpu(_bfaceFlux, boundaryFaces, Float64)
	gamma = ptrToGpu(_gamma, numInteriorFaces, Float64)
	bgamma = ptrToGpu(_bgamma, boundaryFaces, Float64)
	deltaCoeffs = ptrToGpu(_deltaCoeffs, numInteriorFaces, Float64)
	bdeltaCoeffs = ptrToGpu(_bdeltaCoeffs, boundaryFaces, Float64)
	magFaceArea = ptrToGpu(_magFaceArea, numTotalFaces, Float64)
	valueFractions = ptrToGpu(_valueFractions, numTotalFaces, Float64)
	refValue = ptrToGpu(_refValue, boundaryFaces*3, Float64)
	refGradient = ptrToGpu(_refGradient, boundaryFaces*3, Float64)
	RHS = ptrToGpu(_RHS, numCells*3, Float64)
	surfaceCells = ptrToGpu(_surfaceCells, boundaryFaces, Int32)
	bValues= ptrToGpu(_bValues, boundaryFaces*3, Float64)
	bRhs= ptrToGpu(_bRhs, boundaryFaces*3, Float64)
	volumes= ptrToGpu(_volumes, numCells, Float64)
	oldVectors= ptrToGpu(_oldVectors, numCells*3, Float64)
	
	dt = "$_dt"
    opstring2 = replace(opString, "DELTAT" => dt)
    opstring3 = replace(opstring2, "BDF2" => "BDF1")
    fused_pde = eval(Meta.parse(opstring3))

    ddt, spatials = MinimalFVM.splitTempSpat(fused_pde)
	backend = get_backend(vals)
	if !isnothing(ddt)
		ddt_kernel(backend, 64)(
			ddt,
			volumes,
			oldVectors,
			diagOffs,
			rowOffs,
			vals,
			RHS;
			ndrange=numCells
		)		
		KernelAbstractions.synchronize(backend)
	end
	if !isnothing(spatials)
		face_kernel(backend, 64)(
			numInteriorFaces,
			owner,
			neighbour,
			diagOffs,
			ownOffs,
			neiOffs,
			rowOffs,
			vals,
			faceFlux,
			bfaceFlux,
			gamma,
			bgamma,
			deltaCoeffs,
			bdeltaCoeffs,
			magFaceArea,
			valueFractions,
			refValue,
			refGradient,
			RHS,
			surfaceCells,
			bValues,
			bRhs,
			spatials;
			ndrange=numTotalFaces
		)
		KernelAbstractions.synchronize(backend)
	end
end

@kernel function ddt_kernel(
	@Const(temporals),
    @Const(volumes),
    @Const(oldVectors),
    @Const(diagOffs),
    @Const(rowOffs),
    vals,
    RHS
)
	celli = @index(Global)
	idx2D = (celli - 1) * 3 + 1
	valueDiag, rx, ry, rz = temporals(
		oldVectors[idx2D],
		oldVectors[idx2D+1],
		oldVectors[idx2D+2],
		volumes[celli],
		0.0,
		0.0,
		0.0,
		0.0
	)
	idx = (rowOffs[celli] + diagOffs[celli]) * 3 + 1
	vals[idx] += valueDiag
	vals[idx+1] += valueDiag
	vals[idx+2] += valueDiag
	RHS[idx2D] += rx
	RHS[idx2D+1] += rz
	RHS[idx2D+2] += rz
end

@kernel function face_kernel(
    numInteriorFaces,
    @Const(owner),
	@Const(neighbour),
    @Const(diagOffs),
    @Const(ownOffs),    
	@Const(neiOffs),
    @Const(rowOffs),
    vals,
    @Const(faceFlux),  
    @Const(bfaceFlux),  
    @Const(gamma),   
    @Const(bgamma),   
    @Const(deltaCoeffs),
    @Const(bdeltaCoeffs),
    @Const(magFaceArea),
    @Const(valueFractions),
    @Const(refValue),
    @Const(refGradient),
    RHS,    
    @Const(surfaceCells),
    bValues,
    bRhs,
	@Const(fused_pde)
)
	numCells = length(rowOffs) - 1
	iFace = @index(Global)

	if iFace <= numInteriorFaces        
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
        vals[idx]   += valueUpper
        vals[idx+1] += valueUpper
        vals[idx+2] += valueUpper
        
		idx = (rowOwnStart + diagOffs[iOwner]) * 3 + 1
        Atomix.@atomic vals[idx] -= valueUpper
        Atomix.@atomic vals[idx+1] -= valueUpper
        Atomix.@atomic vals[idx+2] -= valueUpper
        
		idx = (rowOwnStart + ownOffs[iFace]) * 3 + 1        
		# vals[idx:idx+2] .+= valueLower
		vals[idx]   += valueLower
        vals[idx+1] += valueLower
        vals[idx+2] += valueLower
        
		idx = (rowNeiStart + diagOffs[iNeighbor]) * 3 + 1
        # FIXME when going back to scalar values
        # vals[idx:idx+2] -= valueLower

        Atomix.@atomic vals[idx]   -= valueLower
        Atomix.@atomic vals[idx+1] -= valueLower
        Atomix.@atomic vals[idx+2] -= valueLower
	else
		bcfacei = iFace - numInteriorFaces
        start = bcfacei * 3 - 2
        fluxContrib, valueRHSx, valueRHSy, valueRHSz = fused_pde(
			refValue[start],
			refValue[start+1],
			refValue[start+2],
			refGradient[start],
			refGradient[start+1],
			refGradient[start+2],
			bfaceFlux[bcfacei],
			valueFractions[bcfacei],
			bdeltaCoeffs[bcfacei],
			bgamma[bcfacei],
			magFaceArea[iFace],
			0.0, 0.0, 0.0, 0.0
        )


		own = surfaceCells[bcfacei] + 1
		vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1

        bValues[bcfacei*3-2] -= fluxContrib
        bValues[bcfacei*3-1] -= fluxContrib
        bValues[bcfacei*3] 	 -= fluxContrib

		Atomix.@atomic vals[vIdx]   -= fluxContrib
        Atomix.@atomic vals[vIdx+1] -= fluxContrib
        Atomix.@atomic vals[vIdx+2] -= fluxContrib
		
        # # FIXME dont forget, changed this back to [xyzxyzxyz] instead of [xxxyyyzzz] for now
        Atomix.@atomic RHS[own*3-2] += valueRHSx
        Atomix.@atomic RHS[own*3-1] += valueRHSy
        Atomix.@atomic RHS[own*3] += valueRHSz
        
        # bRhs[bcfacei*3-2:bcfacei*3] += [valueRHSx, valueRHSy, valueRHSz]
		bRhs[bcfacei*3-2] -= valueRHSx
        bRhs[bcfacei*3-1] -= valueRHSy
        bRhs[bcfacei*3] -= valueRHSz
    end
end

function checkindex(index, arr)
	if index > length(arr)
		@print("vidx: $index, $(length(arr))")
	end
end