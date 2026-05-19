include("common.jl")
using Atomix

function FusedCellBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::PTERM) where {P<:AbstractFloat}
    nu = input.nu
    boundaries = input.boundaries
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement
            valueUpper, valueLower = fused_pde(U[iElement], U[faces.iNeighbor[iFace]], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], zero(P), zero(P))
            idx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            vals[idx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end


function PrecalculatedWeightsCellBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, weights::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    boundaries = input.boundaries
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement

            U_P = U[iElement]
            U_N = U[faces.iNeighbor[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                       # flux through the face
            weights_f = weights[iFace]                      # get precalculated weight
            diffusion = nu[iElement] * faces.gDiff[iFace]          # laplacian(Γ, U)  ⟹ Diffusion
            valueUpper = ϕf * weights_f -diffusion
            valueLower = -ϕf * (1 - weights_f) + diffusion
            
            offdiagIdx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            vals[offdiagIdx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            # convection
            ϕf = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            @inbounds convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
            diag += -diffusion
            # RHS/Source
            value = convection .+ diffusion
            @inbounds rhsx -= value[1]
            @inbounds rhsy -= value[2]
            @inbounds rhsz -= value[3]
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end

function DynamicCellBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, divScheme::Function) where {P<:AbstractFloat}
    nu = input.nu
    boundaries = input.boundaries
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement

            U_P = U[iElement]
            U_N = U[faces.iNeighbor[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                       # flux through the face
            weights_f = divScheme(ϕf)                       # get precalculated weight
            diffusion = nu[iElement] * faces.gDiff[iFace]          # laplacian(Γ, U)  ⟹ Diffusion
            valueUpper = ϕf * weights_f -diffusion
            valueLower = -ϕf * (1 - weights_f) + diffusion
            
            offdiagIdx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            vals[offdiagIdx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            # convection
            ϕf = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            @inbounds convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
            diag += -diffusion
            # RHS/Source
            value = convection .+ diffusion
            @inbounds rhsx -= value[1]
            @inbounds rhsy -= value[2]
            @inbounds rhsz -= value[3]
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end

function HardcodedUpwindCellBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    boundaries = input.boundaries
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement

            U_P = U[iElement]
            U_N = U[faces.iNeighbor[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                       # flux through the face
            weights_f = upwind_f(ϕf)                       # get precalculated weight
            diffusion = nu[iElement] * faces.gDiff[iFace]          # laplacian(Γ, U)  ⟹ Diffusion
            valueUpper = ϕf * weights_f -diffusion
            valueLower = -ϕf * (1 - weights_f) + diffusion
            
            offdiagIdx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            vals[offdiagIdx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            # convection
            ϕf = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            @inbounds convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
            diag += -diffusion
            # RHS/Source
            value = convection .+ diffusion
            @inbounds rhsx -= value[1]
            @inbounds rhsy -= value[2]
            @inbounds rhsz -= value[3]
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end

function HardcodedCDFCellBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    boundaries = input.boundaries
    boundaries = input.boundaries
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement

            U_P = U[iElement]
            U_N = U[faces.iNeighbor[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                       # flux through the face
            weights_f = cdf_f(ϕf)                           # get precalculated weight
            diffusion = nu[iElement] * faces.gDiff[iFace]          # laplacian(Γ, U)  ⟹ Diffusion
            valueUpper = ϕf * weights_f -diffusion
            valueLower = -ϕf * (1 - weights_f) + diffusion
            
            offdiagIdx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            vals[offdiagIdx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            # convection
            ϕf = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            @inbounds convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
            diag += -diffusion
            # RHS/Source
            value = convection .+ diffusion
            @inbounds rhsx -= value[1]
            @inbounds rhsy -= value[2]
            @inbounds rhsz -= value[3]
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end

################
################ Serial
################
################
################
################
################ Parallel
################


function FusedCellBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::PTERM) where {P<:AbstractFloat}
    nu = input.nu
    boundaries = input.boundaries
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    @threads for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement
            valueUpper, valueLower = fused_pde(U[iElement], U[faces.iNeighbor[iFace]], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], zero(P), zero(P))
            idx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            vals[idx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[iElement], faces.gDiff[iFace], diag, rhsx, rhsy, rhsz)
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end


function PrecalculatedWeightsCellBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, weights::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    boundaries = input.boundaries
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    @threads for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement

            U_P = U[iElement]
            U_N = U[faces.iNeighbor[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                       # flux through the face
            weights_f = weights[iFace]                      # get precalculated weight
            diffusion = nu[iElement] * faces.gDiff[iFace]          # laplacian(Γ, U)  ⟹ Diffusion
            valueUpper = ϕf * weights_f -diffusion
            valueLower = -ϕf * (1 - weights_f) + diffusion
            
            offdiagIdx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            Atomix.@atomic vals[offdiagIdx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            # convection
            ϕf = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            @inbounds convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
            diag += -diffusion
            # RHS/Source
            value = convection .+ diffusion
            @inbounds rhsx -= value[1]
            @inbounds rhsy -= value[2]
            @inbounds rhsz -= value[3]
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end

function DynamicCellBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, divScheme::Function) where {P<:AbstractFloat}
    nu = input.nu
    boundaries = input.boundaries
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    @threads for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement

            U_P = U[iElement]
            U_N = U[faces.iNeighbor[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                       # flux through the face
            weights_f = divScheme(ϕf)                       # get precalculated weight
            diffusion = nu[iElement] * faces.gDiff[iFace]          # laplacian(Γ, U)  ⟹ Diffusion
            valueUpper = ϕf * weights_f -diffusion
            valueLower = -ϕf * (1 - weights_f) + diffusion
            
            offdiagIdx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            Atomix.@atomic vals[offdiagIdx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            # convection
            ϕf = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            @inbounds convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
            diag += -diffusion
            # RHS/Source
            value = convection .+ diffusion
            @inbounds rhsx -= value[1]
            @inbounds rhsy -= value[2]
            @inbounds rhsz -= value[3]
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end

function HardcodedUpwindCellBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
    nu = input.nu
    boundaries = input.boundaries
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    @threads for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement

            U_P = U[iElement]
            U_N = U[faces.iNeighbor[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                       # flux through the face
            weights_f = upwind_f(ϕf)                       # get precalculated weight
            diffusion = nu[iElement] * faces.gDiff[iFace]          # laplacian(Γ, U)  ⟹ Diffusion
            valueUpper = ϕf * weights_f -diffusion
            valueLower = -ϕf * (1 - weights_f) + diffusion
            
            offdiagIdx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            Atomix.@atomic vals[offdiagIdx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            # convection
            ϕf = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            @inbounds convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
            diag += -diffusion
            # RHS/Source
            value = convection .+ diffusion
            @inbounds rhsx -= value[1]
            @inbounds rhsy -= value[2]
            @inbounds rhsz -= value[3]
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end

function HardcodedCDFCellBasedAssembly_t(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}) where {P<:AbstractFloat}
        nu = input.nu
    boundaries = input.boundaries
    faces = input.faces
    cells = input.cells
    U_b = input.U_boundary
    U = input.U_internal
    nCells = length(cells.index)
    @threads for iElement in eachindex(cells.index)
        diag, rhsx, rhsy, rhsz = zero(P), zero(P), zero(P), zero(P)
        for localIndex in 1:cells.nInternalFaces[iElement]
            iFace = cells.iFaces[iElement][localIndex]
            isOwner = faces.iOwner[iFace] == iElement

            U_P = U[iElement]
            U_N = U[faces.iNeighbor[iFace]]
            Uf = 0.5(U_P + U_N)                             # interpolate velocity to face 
            ϕf = Uf ⋅ faces.Sf[iFace]                       # flux through the face
            weights_f = cdf_f(ϕf)                           # get precalculated weight
            diffusion = nu[iElement] * faces.gDiff[iFace]          # laplacian(Γ, U)  ⟹ Diffusion
            valueUpper = ϕf * weights_f -diffusion
            valueLower = -ϕf * (1 - weights_f) + diffusion
            
            offdiagIdx = ifelse(isOwner, faces.ownerRelOwnerIdx[iFace], faces.neighborRelNeighborIdx[iFace])
            Atomix.@atomic vals[offdiagIdx] += ifelse(isOwner, valueLower, valueUpper)
            diag += ifelse(!isOwner, valueLower, valueUpper)
        end
        for iFace in cells.iFaces[iElement][cells.nInternalFaces[iElement]+1:end]
            iBoundary = faces.patchIndex[iFace]
            boundaryType = U_b[iBoundary].type
            if boundaryType != "fixedValue"
                continue
            end
            relativeFaceIndex = iFace - boundaries[iBoundary].startFace
            # convection
            ϕf = faces.Sf[iFace] ⋅ U_b[iBoundary].values[relativeFaceIndex]
            @inbounds convection = U_b[iBoundary].values[relativeFaceIndex] .* ϕf
            # diffusion 
            diffusion = nu[faces.iOwner[iFace]] * faces.gDiff[iFace]
            diag += -diffusion
            # RHS/Source
            value = convection .+ diffusion
            @inbounds rhsx -= value[1]
            @inbounds rhsy -= value[2]
            @inbounds rhsz -= value[3]
        end
        RHS[iElement] += rhsx
        RHS[iElement+nCells] += rhsy
        RHS[iElement+nCells+nCells] += rhsz
        vals[cells.rowOffset[iElement]] += diag
    end
    return vals, RHS
end
