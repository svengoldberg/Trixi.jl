###################################################################################################
# Script used to simulate a dam break problem over a B-spline interpolated section of the         #
# Rhine river valley                                                                              #
###################################################################################################

using LinearAlgebra
using OrdinaryDiffEq
using Trixi

# Get root directory
dir_path = pkgdir(Trixi) # Moved rhine_data_file into Trixi.jl folder

# Define data path
Rhine_data = string(dir_path, "/examples/TrixiBottomTopography/data/rhine_data_2d_20.txt")

# B-spline interpolation of the underlying data
spline_struct = BicubicBSpline(Rhine_data)
spline_func(x,y) = spline_interpolation(spline_struct, x, y)

equations = ShallowWaterEquations2D(gravity_constant=9.81, H0=60.0)

#=
  Currently, error-based time step is chosen
  Tolerances have to be relatively large to not produce time steps around 1e-5
  But nevertheless, results looks very good and high resolution (Ref5, LGL6 e.g.) is achieved
=#

function initial_condition_wave(x, t, equations::ShallowWaterEquations2D)

  x1, x2 = x
  b = spline_func(x1, x2)

  v1 = 0.0
  v2 = 0.0

  inicenter = SVector(31.0, 27.0)
  x_norm = x - inicenter
  r = norm(x_norm)

  # Calculate primitive variables
  # use a logistic function to tranfer water height value smoothly
  L  = equations.H0    # maximum of function
  x0 = 7.0   # center point of function
  k  = -1.0 # sharpness of transfer
  
  # Cylinder
  H = max(b, L/(1.0 + exp(-k*(sqrt((x1-inicenter[1])^2+(x2-inicenter[2])^2) - x0))))
  # Larger water column
  #H = max(b, L/(1.0 + exp(-k*(sqrt((x1-inicenter[1])^2 + 1.25*(x1-inicenter[1])*(x2-inicenter[2]) + 0.5*(x2-inicenter[2])^2) - x0))))

  H = max(H, b + equations.threshold_limiter)
  return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_wave
boundary_condition = BoundaryConditionDirichlet(initial_condition)

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)

surface_flux = (FluxHydrostaticReconstruction(flux_hll_cn, hydrostatic_reconstruction_chen_noelle),
                flux_nonconservative_chen_noelle)

basis = LobattoLegendreBasis(6)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.6,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = (spline_struct.x[1], spline_struct.y[1])
coordinates_max = (spline_struct.x[end], spline_struct.y[end])

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=10_000,
                periodicity=false
               )

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                   boundary_conditions=boundary_condition)
                                   
###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 15.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=cfl)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(equations.threshold_limiter,),
                                                     variables=(Trixi.waterheight,))

sol = solve(ode, SSPRK43(stage_limiter!),
            dt=1.0, abstol=1.0e-3, reltol=1.0e-3,
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
