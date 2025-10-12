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
    lines = readlines(TFileName)[19:(end)]
    split18 = match(r"(\w+)\s+(\w+)\s(\d+);", lines[1])
    if split18[1] == "internalField" && split18[2] == "uniform"
        value = tryparse(Int, split18[3])
        internalTemperatureField.values = [value for x in internalTemperatureField.values]
    end
    boundaryTemperatureFields = [BoundaryField(0, [], "") for _ in 1:(size(mesh.boundaries)[1])]

    if occursin("boundaryField", lines[3])
        pointer = 5
        for (index, boundary) in enumerate(mesh.boundaries)
            name = strip(lines[pointer])
            if name != boundary.name
                continue
            end
            boundaryTemperatureFields[index].values = zeros(boundary.nFaces)
            boundaryTemperatureFields[index].nFaces = boundary.nFaces
            pointer += 2
            while !occursin("}", lines[pointer])
                vals = match(r"(\w+)\s+(\w+)\s?(\d+)?;", lines[pointer])
                if vals[1] == "type"
                    boundaryTemperatureFields[index].type = vals[2]
                    if vals[2] == "empty"
                        boundaryTemperatureFields[index].values = []
                        boundaryTemperatureFields[index].nFaces = 0
                    end
                elseif vals[1] == "value" && vals[2] == "uniform"
                    temp = tryparse(Float64, vals[3])
                    boundaryTemperatureFields[index].values = [temp for x in boundaryTemperatureFields[index].values]
                end
                pointer += 1
            end
            pointer += 1
        end
    end
    return boundaryTemperatureFields
end # function readTemperatureField

export readTemperatureField, heatInduction
end # module HeatInduction