module HeatInduction
using ArgParse
include("MeshIO.jl")
# include("Structs.jl")
using LinearAlgebra
using .MeshIO
using .MeshIO.Structs

arg_info = ArgParseSettings()
@add_arg_table arg_info begin
	"--case_directory"
	arg_type = String
	required = true
	"--assembly_method"
	arg_type = String
	required = true
	help = "<cell/face/batchedFace>"

end

if abspath(PROGRAM_FILE) == @__FILE__
	args = parse_args(arg_info)
	case_directory = args["case_directory"]
	assembly_method = args["assembly_method"]
	try
		main(case_directory)
	catch error
		println(error)
	end
end


function main(caseDirectory::String)
	mesh = MeshIO.readOpenFoamMesh(caseDirectory)
	# Define the thermal conductivity and source term
	thermalConductivity = ones(size(mesh.faces)[1])
	numCells = size(mesh.cells)[1]
	numBoundaries = size(mesh.boundaries)[1]
	heatSource = zeros(numCells)

	# Read initial condition and boundary conditions

	internalTemperatureField = Field(numCells, zeros(numCells))
	boundaryTemperatureFields = readTemperatureField(caseDirectory, mesh, internalTemperatureField)

    println(mesh)
    # Assemble the coefficient matrix and RHS vector
    matrix, RHS = cellBasedAssembly(mesh, heatSource, thermalConductivity, boundaryTemperatureFields)
end # function main

function cellBasedAssembly(mesh::Mesh, source::Vector{Float64}, diffusionCoeff::Vector{Float64}, boundaryFields::Vector{BoundaryField})::Tuple{Matrix{Float64}, Vector{Float64}}
    nCells = size(mesh.cells)[1]
    RHS = zeros(nCells)
    coeffMatrix = Matrix(zeros(nCells, nCells))
    println(coeffMatrix)
    # for (iElement, theElement) in enumerate(mesh.cells)
    for iElement in 1:nCells
        theElement = mesh.cells[iElement]
        RHS[iElement] = source[iElement] * theElement.volume
        diag = 0.0
        nFaces = size(theElement.iFaces)[1]
        println("numfaces: $nFaces")
        for (iFace, iFaceIndex) in enumerate(theElement.iFaces)
            theFace = mesh.faces[iFaceIndex]
            if theFace.iNeighbor != -1 # if interior face
                println("$(theFace.index) is interior")
                FluxCn = diffusionCoeff[iFaceIndex] * theFace.gDiff
                FluxFn = -FluxCn
                println("cell: $(iElement), iFace: $(iFace), nb: $(theElement.iNeighbors)")
                coeffMatrix[iElement, theElement.iNeighbors[iFace]] = FluxFn
                diag += FluxCn
            else # if boundary face
                iBoundary = mesh.faces[iFaceIndex].patchIndex
                println("$(theFace.index) is exterior mit patchindex $(iBoundary)")
                boundaryType = boundaryFields[iBoundary].type
                FluxCb = 0.0
                FluxVb = 0.0       
                if boundaryType == "fixedValue" # dirichlet BC
                    println("fixedValue for $(theFace.index)")
                    FluxCb = diffusionCoeff[iFaceIndex] * theFace.gDiff
                    relativeFaceIndex = iFaceIndex - mesh.boundaries[iBoundary].startFace
                    println("$relativeFaceIndex = $iFaceIndex - $(mesh.boundaries[iBoundary].startFace)")
                    FluxVb = -FluxCb * boundaryFields[iBoundary].values[relativeFaceIndex]
                    diag += FluxCb
                    RHS[iElement] -= FluxCb
                else 
                    # ignore zeroGradient and empty
                    # Do nothing because FluxCb and FluxVb are already 0.0
                    # Do nothing because the face does not contribute
                end
            end
        end
        coeffMatrix[iElement, iElement] = diag
    end
    return coeffMatrix, RHS
end # function cellBasedAssembly


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
            println("$(name), $(boundary.name)")
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
                    println("iBoundary: $index, type: $(vals[2])")
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


end # module HeatInduction
