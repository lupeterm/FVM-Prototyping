include("../init.jl")
include("../cpu_helper.jl")
upwind_f(flux) = ifelse(flux >= 0, 1.0, 0.0)
cdf_f(flux) = 0.5