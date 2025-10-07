
using ArgParse
using HeatInduction
arg_info = ArgParseSettings()
@add_arg_table arg_info begin
	"--case_directory"
	arg_type = String
	required = true
	"--assembly_method"
	arg_type = String
	default = "cell"
	help = "<cell/face/batchedFace>"

end

if abspath(PROGRAM_FILE) == @__FILE__
	args = parse_args(arg_info)
	case_directory = args["case_directory"]
	assembly_method = args["assembly_method"]
	try
		heatInduction(case_directory)
	catch error
		println(error)
	end
end

