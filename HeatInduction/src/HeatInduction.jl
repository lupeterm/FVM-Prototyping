module HeatInduction
using ArgParse
include("MeshIO.jl")
using .MeshIO

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
        MeshIO.readOpenFoamMesh(case_directory)
    catch error
        println(error)
    end
end


end # module HeatInduction
