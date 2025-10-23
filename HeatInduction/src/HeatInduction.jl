module HeatInduction
# include("MeshStructs.jl")
using MeshStructs
include("ReadMesh.jl")

function heatInduction(caseDirectory::String)::MatrixAssemblyInput
    mesh = readOpenFoamMesh(caseDirectory)
    # Define the thermal conductivity and source term
    thermalConductivity = ones(size(mesh.faces)[1])
    numCells = size(mesh.cells)[1]
    heatSource = zeros(numCells)

    # Read initial condition and boundary conditions
    boundaryTemperatureFields = readField(joinpath(caseDirectory, "0/T"), mesh) 

    # Assemble the coefficient matrix and RHS vector
    matrixAssemblyInput = MatrixAssemblyInput(mesh, heatSource, thermalConductivity, boundaryTemperatureFields)
    return matrixAssemblyInput
end # function heatInduction

function genericHeatInduction(caseDirectory::String)::GenericMatrixAssemblyInput
    mesh = readOpenFoamMesh(caseDirectory)
    # Define the thermal conductivity and source term
    thermalConductivity = ones(size(mesh.faces)[1])
    numCells = size(mesh.cells)[1]
    heatSource = zeros(numCells)

    # Read initial condition and boundary conditions
    boundaryTemperatureFields = readField(joinpath(caseDirectory, "0/T"), mesh) 
    mapping = Dict()
    mapping[1] = "T"
    # Assemble the coefficient matrix and RHS vector
    matrixAssemblyInput = GenericMatrixAssemblyInput(mesh, [heatSource], [thermalConductivity], boundaryTemperatureFields, mapping)
    return matrixAssemblyInput
end # function heatInduction

function LidDrivenCavity(caseDirectory::String)::LdcMatrixAssemblyInput
    mesh = readOpenFoamMesh(caseDirectory)
    # Define the thermal conductivity and source term
    nu = readPropertiesFile(joinpath(caseDirectory, "constant/transportProperties"))
    # Read initial condition and boundary conditions
    p = readField(joinpath(caseDirectory, "0/p"), mesh)
    U = readField(joinpath(caseDirectory, "0/U"), mesh)

    # Assemble the coefficient matrix and RHS vector
    ldcInput = LdcMatrixAssemblyInput(mesh, nu, p, U)
    return ldcInput
end # function LidDrivenCavity


export heatInduction, genericHeatInduction, LidDrivenCavity
end # module HeatInduction