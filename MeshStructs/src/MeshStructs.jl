module MeshStructs
using StaticArrays

struct CaseDirError <: Exception
    message::String
end # struct CaseDirError



mutable struct Face
    index::Int
    iNodes::Vector{Int}
    iOwner::Int
    iNeighbor::Int
    centroid::MVector{3, Float32}
    Sf::MVector{3, Float32}
    area::Float32
    CN::Vector{Float32}
    magCN::Float32
    eCN::Vector{Float32}
    gDiff::Float32
    T::Vector{Float32}
    gf::Float32
    patchIndex::Int
    walldist::Float32
    iOwnerNeighborCoef::Int
    iNeighborOwnerCoef::Int
end # struct Face

mutable struct Node
    centroid::SVector{3, Float32}
    iCells::Vector{Int}
    iFaces::Vector{Int}
    flag::Int
end # struct Node

struct Boundary
    name::String
    type::String
    inGroups::Tuple{Int,String}
    nFaces::Int
    startFace::Int
end # struct Boundary

mutable struct Cell
    index::Int
    iFaces::Vector{Int}
    iNeighbors::Vector{Int}
    numNeighbors::Int
    faceSigns::Vector{Int}
    iNodes::Vector{Int}
    volume::Float32
    oldVolume::Float32
    centroid::Vector{Float32}
end # struct Cell

struct Mesh
    nodes::Vector{Node}
    faces::Vector{Face}
    boundaries::Vector{Boundary}
    numCells::Int
    cells::Vector{Cell}
    numInteriorFaces::Int
    numBoundaryCells::Int
    numBoundaryFaces::Int
end # struct Mesh

mutable struct Field{T}
    nElements::Int
    values::Vector{T}
end # struct Field

mutable struct BoundaryField{T}
    name::String
    nFaces::Int
    values::Vector{T}
    type::String
end # struct BoundaryField

import Base: show

function show(io::IO, node::Node)
    printNode(node)
end

function show(io::IO, cell::Cell)
    printCell(cell)
end

function show(io::IO, face::Face)
    printFace(face)
end

function show(io::IO, boundary::Boundary)
    printBoundary(boundary)
end

function show(io::IO, mesh::Mesh)
    printMesh(mesh)
end

function printMesh(mesh::Mesh)
    println("Mesh:")
    println("  Number of Cells: ", mesh.numCells)
    println("  Number of Interior Faces: ", mesh.numInteriorFaces)
    println("  Number of Boundary Cells: ", mesh.numBoundaryCells)
    println("  Number of Boundary Faces: ", mesh.numBoundaryFaces)

    println("  Nodes: [")
    for (i, node) in enumerate(mesh.nodes)
        println("    Node[$i]:")
        println(node)
    end
    println("  ]")

    println("  Faces: [")
    for (i, face) in enumerate(mesh.faces)
        println(face) 
    end
    println("  ]")

    println("  Boundaries: [")
    for (i, boundary) in enumerate(mesh.boundaries)
        println(boundary) 
    end
    println("  ]")

    println("  Cells: [")
    for (i, cell) in enumerate(mesh.cells)
        println(cell)
    end
    println("  ]")
end


function printFace(face::Face)
    println("\tFace $(face.index) {")
    println("\t\tnNodes: ", size(face.iNodes)[1])
    println("\t\tiNodes: ", face.iNodes)
    println("\t\tiOwner: ", face.iOwner)
    println("\t\tiNeighbor: ", face.iNeighbor)
    println("\t\tcentroid: ", face.centroid)
    println("\t\tSf: ", face.Sf)
    println("\t\tarea: ", face.area)
    println("\t\tmagCN: ", face.magCN)
    println("\t\tCN: ", face.CN)
    println("\t\teCN: ", face.eCN)
    println("\t\tgDiff: ", face.gDiff)
    println("\t\tT: ", face.T)
    println("\t\tgf: ", face.gf)
    println("\t\tpatchIndex: ", face.patchIndex)
    println("\t}")
end

function printBoundary(b::Boundary)
    println("\tBoundary {")
    println("\t\tname: ", b.name)
    println("\t\ttype: ", b.type)
    println("\t\tinGroups: (", b.inGroups[1], ", \"", b.inGroups[2], "\")")
    println("\t\tnFaces: ", b.nFaces)
    println("\t\tstartFace: ", b.startFace)
    println("\t}")
end

function printNode(node::Node)
    println("\tNode:")
    println("\t\tCentroid: ", node.centroid)
    println("\t\tiCells: ", node.iCells)
    println("\t\tiFaces: ", node.iFaces)
    println("\t\tFlag: ", node.flag)
end

function printCell(cell::Cell)
    println("\tCell $(cell.index):")
    println("\t\tiFaces: ", cell.iFaces)
    println("\t\tiNeighbors: ", cell.iNeighbors)
    println("\t\tNumNeighbors: ", cell.numNeighbors)
    println("\t\tFaceSigns: ", cell.faceSigns)
    println("\t\tiNodes: ", cell.iNodes)
    println("\t\tVolume: ", cell.volume)
    println("\t\tOldVolume: ", cell.oldVolume)
    println("\t\tCentroid: ", cell.centroid)
end

struct MatrixAssemblyInput{T}
    mesh::Mesh
    source::Vector{Float32}
    diffusionCoeff::Vector{Float32}
    boundaryFields::Vector{BoundaryField{T}}
end

struct LdcMatrixAssemblyInput
    mesh::Mesh
    nu::Float32
    p::Vector{BoundaryField}
    U::Vector{BoundaryField}
end

struct GenericMatrixAssemblyInput
    mesh::Mesh
    sources::Vector{Vector{Float32}}
    variables::Vector{Float32}
    boundaryFields::Vector{Vector{BoundaryField}}
    mappings::Dict{Int, String}
end

export CaseDirError,
    Face,
    Node,
    Boundary,
    Cell,
    Mesh,
    Field,
    BoundaryField,
    printBoundary,
    printFace,
    printCell,
    printNode,
    MatrixAssemblyInput,
    LdcMatrixAssemblyInput,
    GenericMatrixAssemblyInput

end # module MeshStructs
