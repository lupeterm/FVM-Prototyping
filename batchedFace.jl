include("init.jl")



function getGreedyEdgeColoring(input::MatrixAssemblyInput)
    mesh = input.mesh
    for face::Face in mesh.faces
        face.batchId = -1
    end
    faceColorMapping = zeros(Int32, mesh.numInteriorFaces)
    for cell::Cell in mesh.cells
        usedColors = []
        for iFace in cell.iFaces[1:cell.nInternalFaces]
            face::Face = mesh.faces[iFace]
            if face.batchId == -1
                continue
            end
            push!(usedColors, face.batchId)
        end
        id = 1
        for iFace in cell.iFaces[1:cell.nInternalFaces]
            face::Face = mesh.faces[iFace]
            if face.batchId != -1
                continue
            end
            while true
                if id in usedColors
                    id += 1
                    continue
                end
                face.batchId = id
                faceColorMapping[face.index] = id
                push!(usedColors, face.batchId)
                break
            end
        end
    end
    return maximum(faceColorMapping), faceColorMapping
end