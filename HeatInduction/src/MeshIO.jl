module MeshIO

struct CaseDirError <: Exception
	message::String
end # struct CaseDirError

struct Node
	x::Float32
	y::Float32
	z::Float32
end # struct Node

struct Face
	nodes::Vector{Int}
end # struct Face

struct Boundary
	name::String
	type::String
	inGroups::Tuple{Int, String}
	nFaces::Int
	startFace::Int
end

function readOpenFoamMesh(caseDir::String)
	if !isdir(caseDir)
		throw(CaseDirError("Case Directory '$(caseDir)' does not exist"))
	end
	polymeshDir = joinpath(caseDir, "constant/polyMesh")
	if !isdir(polymeshDir)
		throw(CaseDirError("PolyMesh Directory '$(caseDir)' does not exist"))
	end
	println("Reading OpenFOAM mesh files from mesh directory: $caseDir")

	nodes = readPointsFile(polymeshDir)
	faces = readFacesFile(polymeshDir)
	owner = readOwnersFile(polymeshDir)
	neighbors = readNeighborsFile(polymeshDir);
	boundaries = readBoundaryFile(polymeshDir);
	println(boundaries)
	#   cellsconstructCells(fvMesh);
	#   setupNodeConnectivities(fvMesh);
	#   MeshProcessor.processOpenFoamMesh(fvMesh);
end # function readOpenFoamMesh



function readPointsFile(polyMeshDir::String)::Vector{Node}
	pointsFileName = joinpath(polyMeshDir, "points")
	if !isfile(pointsFileName)
		throw(CaseDirError("Points file '$(pointsFileName)' does not exist."))
	end
	lines = readlines(pointsFileName)[22:(end-4)]
	nodes::Vector{Node} = []
	for line in lines
		s = split((line[2:(end-1)]), " ")
		coords = map(s -> tryparse(Float32, s), s)
		push!(nodes, Node(coords...))
	end
	return nodes

end # function readPointsFile

function readFacesFile(polyMeshDir::String)::Vector{Face}
	facesFileName = joinpath(polyMeshDir, "faces")
	if !isfile(facesFileName)
		throw(CaseDirError("Faces file '$(facesFileName)' does not exist."))
	end
	lines = readlines(facesFileName)[22:(end-4)]
	faces::Vector{Face} = []
	for line in lines
		s = split((line[3:(end-1)]), " ")
		nodes = map(s -> tryparse(Int, s), s)
		push!(faces, Face(nodes))
	end
	return faces
end # function readPointsFile

function readOwnersFile(polyMeshDir::String)::Vector{Int}
	ownersFileName = joinpath(polyMeshDir, "owner")
	if !isfile(ownersFileName)
		throw(CaseDirError("Owners file '$(ownersFileName)' does not exist."))
	end
	lines = readlines(ownersFileName)[23:(end-4)]
	owners = map(l -> tryparse(Int, l), lines)
	return owners
end # function readPointsFile


function readNeighborsFile(polyMeshDir::String)::Vector{Int}
	neighborsFileName = joinpath(polyMeshDir, "neighbour")
	if !isfile(neighborsFileName)
		throw(CaseDirError("Neighbors file '$(neighborsFileName)' does not exist."))
	end
	line = readlines(neighborsFileName)[20]
	s = split((line[3:(end-1)]), " ")
	nbs = map(s -> tryparse(Int, s), s)
	return nbs
end # function readPointsFile

function readBoundaryFile(polyMeshDir::String)::Vector{Boundary}
	boundariesFileName = joinpath(polyMeshDir, "boundary")
	if !isfile(boundariesFileName)
		throw(CaseDirError("Boundary file '$(boundariesFileName)' does not exist."))
	end

	boundaries::Vector{Boundary} = []
	lines = readlines(boundariesFileName)[21:(end-4)]
	l = split(join(lines), "}")
	for bound in l
		s = replace(bound, r"\s+" => " ")
		s = replace(bound, ";" => "")
		s = split(s, " ")
		s = filter(x -> x != "", s)
		ing1 = tryparse(Int, "$(s[6][1])")
		ing = (ing1, s[6][3:(end-1)])
		boundary = Boundary(
			s[1],
			s[4],
			ing,
			tryparse(Int, s[8]),
			tryparse(Int, s[10]),
		)
		push!(boundaries, boundary)
	end
	return boundaries
end # function readPointsFile

function constructCells()
    # TODO
end

end # module MeshIO
