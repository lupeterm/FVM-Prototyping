
module MeshProcessor
using MeshStructs
using LinearAlgebra

function processOpenFoamMesh(mesh::Mesh)::Mesh
    mesh = processBasicFaceGeometry(mesh)
    mesh = computeElementVolumeAndCentroid(mesh)
    mesh = processSecondaryFaceGeometry(mesh)
    mesh = sortBoundaryNodesFromInteriorNodes(mesh)
    mesh = labelBoundaryFaces(mesh)
    return mesh
end # function processOpenFoamMesh

function magnitude(vector::Vector{Float64})::Float64
    return sqrt(dot(vector, vector))
end # function magnitude

function processBasicFaceGeometry(mesh::Mesh)::Mesh
    println("Processing Mesh Geometry")
    for (index, face) in enumerate(mesh.faces)
        centroid = zeros(3)
        Sf = zeros(3)
        area = 0.0
        # special case: triangle
        if size(face.iNodes)[1] == 3
            # sum x,y,z and divide by 3
            triangleNodes = map(n -> n.centroid, mesh.nodes[face.iNodes])
            centroid = sum(triangleNodes) / 3
            Sf = 0.5 * cross(triangleNodes[2] - triangleNodes[1], triangleNodes[3] - triangleNodes[1])
            area = magnitude(Sf)
        else # general case, polygon is not a triangle 
            nodes = map(n -> n.centroid, mesh.nodes[face.iNodes])
            center = sum(nodes) / size(nodes)[1]
            # Using the center to compute the area and centroid of virtual
            # triangles based on the center and the face nodes
            triangleNode1 = center
            triangleNode2 = zeros(3)
            triangleNode3 = zeros(3)
            for (index, iNode) in enumerate(face.iNodes)
                triangleNode2 = mesh.nodes[iNode].centroid
                if index < size(face.iNodes)[1] - 1
                    triangleNode3 = mesh.nodes[face.iNodes[index+1]].centroid
                else
                    triangleNode3 = mesh.nodes[face.iNodes[1]].centroid
                end
                # Calculate the centroid of a given subtriangle
                localCentroid = (triangleNode1 + triangleNode2 + triangleNode3) / 3
                # Calculate the surface area vector of a given subtriangle by cross product
                localSf = 0.5 * cross(triangleNode2 - triangleNode1, triangleNode3 - triangleNode2)
                # Calculate the surface area of a given subtriangle
                localArea = sqrt(dot(localSf, localSf))
                centroid += localArea * localCentroid
                Sf += localSf
            end
            area = magnitude(Sf)
            # Compute centroid of the polygon
            centroid /= area
        end
        face.centroid = centroid
        face.Sf = Sf
        face.area = area
    end
    return mesh
end # function processBasicFaceGeometry

function computeElementVolumeAndCentroid(mesh)::Mesh
    println("Computing Element Volumen and Centroid")
    for cell in mesh.cells
        elementCenter = zeros(3)
        elementCenter = sum(map(f -> f.centroid, mesh.faces[cell.iFaces])) / size(cell.iFaces)[1]

        # Compute volume and centroid of each element
        elementCentroid = zeros(3)
        localVolumeCentroidSum = zeros(3)
        localVolumeSum = 0.0
        for i in 1:size(cell.iFaces)[1]
            localFace = mesh.faces[i]
            localFaceSign = cell.faceSigns[i]
            Sf = localFaceSign * localFace.Sf

            # Calculate the distance vector from geometric center to the face centroid
            d_Gf = localFace.centroid - elementCenter

            # Calculate the volume of each sub-element pyramid
            localVolume = dot(Sf, d_Gf) / 3.0
            localVolumeSum += localVolume

            #Calculate volume-weighted center of sub-element pyramid (centroid)
            localCentroid = localFace.centroid * 0.75 + 0.25 * elementCenter
            localVolumeCentroidSum += localCentroid * localVolume
        end
        cell.centroid = (1 / localVolumeSum)' * localVolumeCentroidSum
        cell.volume = localVolumeSum
        cell.oldVolume = localVolumeSum
    end
    return mesh
end # function computeElementVolumeAndCentroid

function processSecondaryFaceGeometry(mesh::Mesh)::Mesh
    # Loop over interior faces
    for face in mesh.faces[1:mesh.numInteriorFaces]
        # Compute unit surface normal vector
        nf = (1 / face.area) * face.Sf
        ownerElement = mesh.cells[face.iOwner]
        neighborElement = mesh.cells[face.iNeighbor]
        face.CN = neighborElement.centroid - ownerElement.centroid

        face.magCN = magnitude(face.CN)
        face.eCN = (1 / face.magCN) * face.CN

        E = face.area * face.eCN

        face.gDiff = magnitude(E) / face.magCN

        face.T = face.Sf - E

        # Compute face weighting factor
        Cf = face.centroid - ownerElement.centroid
        fF = neighborElement.centroid - face.centroid
        face.gf = dot(Cf, nf) / (dot(Cf, nf) + dot(fF, nf))
    end
    return mesh
end # function processSecondaryFaceGeometry

function sortBoundaryNodesFromInteriorNodes(mesh::Mesh)::Mesh
    for face in mesh.faces[1:mesh.numInteriorFaces]
        for iNode in face.iNodes
            mesh.nodes[iNode].flag = 1
        end
    end
    for boundary in mesh.boundaries
        startFace = boundary.startFace
        nBFaces = boundary.nFaces
        s1 = boundary.name == "frontAndBack"
        s2 = boundary.name == "frontAndBackPlanes"
        for face in mesh.faces[startFace:(startFace+nBFaces)]
            for iNode in face.iNodes
                mesh.nodes[iNode].flag = (s1 || s2) ? 1 : 0
            end
        end
    end
    return mesh
end # function sortBoundaryNodesFromInteriorNodes

function labelBoundaryFaces(mesh::Mesh)::Mesh
    for (iBoundary, boundary) in enumerate(mesh.boundaries)
        startFace = boundary.startFace+1
        nBFaces = boundary.nFaces
        println("startface for iBoundary $iBoundary is $startFace, nbfaces : $nBFaces")
        for iFace in startFace:(startFace+nBFaces-1)
            println("setting $iBoundary as pindex for $(mesh.faces[iFace].index)")
            mesh.faces[iFace].patchIndex = iBoundary
        end
    end
    return mesh
end # function labelBoundaryFaces
export processOpenFoamMesh
end 