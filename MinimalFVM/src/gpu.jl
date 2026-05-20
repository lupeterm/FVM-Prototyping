using CUDA
using KernelAbstractions
using Atomix
using StaticArrays

function ptrToGpu(ptr::Ptr{Cvoid}, size, dType)
	p = reinterpret(CuPtr{dType}, ptr)
    arr = unsafe_wrap(CuArray, p, Int64(size); own=false)
	return arr
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
	_gamma::Ptr{Cvoid},
	_deltaCoeffs::Ptr{Cvoid},
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
	faceFlux = ptrToGpu(_faceFlux, numTotalFaces, Float64)
	gamma = ptrToGpu(_gamma, numTotalFaces, Float64)
	deltaCoeffs = ptrToGpu(_deltaCoeffs, numInteriorFaces, Float64)
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
			RHS
			;
			ndrange=numCells
		)		
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
			gamma,
			deltaCoeffs,
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
	idx = rowOffs[celli] + 1 + diagOffs[celli]
	vals[idx] = valueDiag
	RHS[idx2D] = rx
	RHS[idx2D+1] = rz
	RHS[idx2D+2] = rz
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
    @Const(gamma),   
    @Const(deltaCoeffs),
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
        end_ = start + 2
        # helperSVector1 = SVector{3,Float64}(refValue[start:end_])
        # helperSVector2 = SVector{3,Float64}(refGradient[start:end_])
        valueDiag, valueRHSx, valueRHSy, valueRHSz = fused_pde(
			refValue[start],
			refValue[start+1],
			refValue[start+2],
			refGradient[start],
			refGradient[start+1],
			refGradient[start+2],
			faceFlux[iFace],
			valueFractions[bcfacei],
			6.0,
			3.0,
			gamma[iFace],
			magFaceArea[iFace],
			0.0, 0.0, 0.0, 0.0
        )

		own = surfaceCells[bcfacei] + 1

        vIdx = (rowOffs[own] + diagOffs[own]) * 3 + 1

		Atomix.@atomic vals[vIdx]   += valueDiag
        Atomix.@atomic vals[vIdx+1] += valueDiag
        Atomix.@atomic vals[vIdx+2] += valueDiag
		
        bValues[bcfacei*3-2] = valueDiag
        bValues[bcfacei*3-1] = valueDiag
        bValues[bcfacei*3] 	 = valueDiag


        # FIXME dont forget, changed this back to [xyzxyzxyz] instead of [xxxyyyzzz] for now
        Atomix.@atomic RHS[own*3-2] += valueRHSx
        Atomix.@atomic RHS[own*3-1] += valueRHSy
        Atomix.@atomic RHS[own*3] += valueRHSz
        
		RHS[own*3-2] += valueRHSx
        RHS[own*3-1] += valueRHSy
        RHS[own*3] += valueRHSz

        # bRhs[bcfacei*3-2:bcfacei*3] += [valueRHSx, valueRHSy, valueRHSz]
		Atomix.@atomic bRhs[bcfacei*3-2] += valueRHSx
        Atomix.@atomic bRhs[bcfacei*3-1] += valueRHSy
        Atomix.@atomic bRhs[bcfacei*3] += valueRHSz
    end
end

function checkindex(index, arr)
	if index > length(arr)
		@print("vidx: $index, $(length(arr))")
	end
end