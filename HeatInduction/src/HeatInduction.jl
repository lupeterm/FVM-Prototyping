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

    internalTemperatureField = Field(numCells, zeros(numCells))
    boundaryTemperatureFields = readTemperatureField(caseDirectory, mesh, internalTemperatureField)
    # Assemble the coefficient matrix and RHS vector
    matrixAssemblyInput = MatrixAssemblyInput(mesh, heatSource, thermalConductivity, boundaryTemperatureFields)
    return matrixAssemblyInput
end # function main

function readTemperatureField(caseDir::String, mesh::Mesh, internalTemperatureField::Field)::Vector{BoundaryField}
    TFileName = joinpath(caseDir, "0/T")
    if !isfile(TFileName)
        throw(CaseDirError("T file '$(TFileName)' does not exist."))
    end
    i = 17
    lines = readlines(TFileName)
    while !startswith(lines[i], "internalField")
        i += 1
    end
    split18 = match(r"(\w+)\s+(\w+)\s(\d+);", lines[i])
    if split18[1] == "internalField" && split18[2] == "uniform"
        value = tryparse(Int, split18[3])
        internalTemperatureField.values = [value for x in internalTemperatureField.values]
    end
    while !contains(lines[i], "boundaryField")
        i += 1
    end
    boundaryLines = lines[i+2:end-4]
    joined = join(boundaryLines)
    splitted = split(joined, "}")
    boundaryTemperatureFields = [BoundaryField(0, [], "") for _ in 1:(size(mesh.boundaries)[1])]
    for (index, b) in enumerate(splitted)
        matches = match(r"\s*(\w+)\s*\{\s*type\s*(\w+);(?:\s*value\s*(\w+) (\d+);)?", b)
        boundaryTemperatureFields[index].values = zeros(mesh.boundaries[index].nFaces)
        boundaryTemperatureFields[index].nFaces = mesh.boundaries[index].nFaces
        boundaryTemperatureFields[index].type = matches[2]
        if matches[2] == "empty"
            boundaryTemperatureFields[index].values = []
            boundaryTemperatureFields[index].nFaces = 0
        end
        if !isnothing(matches[3])
            temp = tryparse(Float64, matches[4])
            boundaryTemperatureFields[index].values = fill(temp, boundaryTemperatureFields[index].nFaces) 
        end
    end
    return boundaryTemperatureFields
end # function readTemperatureField

export readTemperatureField, heatInduction
end # module HeatInduction