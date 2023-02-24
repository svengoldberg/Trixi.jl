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
city_data = string(dir_path, "/examples/TrixiBottomTopography/data/seaside_oregon.txt")

# B-spline interpolation of the underlying data
spline_struct = BicubicBSpline(city_data)
spline_func(x,y) = spline_interpolation(spline_struct, x, y)

equations = ShallowWaterEquations2D(gravity_constant=9.81, H0=1.0)


function initial_condition_wave(x, t, equations::ShallowWaterEquations2D)

  x1, x2 = x
  b = spline_func(x1, x2)

  inicenter = SVector(31.0, 27.0)
  x_norm = x - inicenter
  r = norm(x_norm)

  # Calculate primitive variables
  # use a logistic function to tranfer water height value smoothly
  L  = equations.H0    # maximum of function
  x0 = 7.0   # center point of function
  k  = -1.0 # sharpness of transfer
  
  #H = equations.H0 #max(b, L/(1.0 + exp(-k*(sqrt((x1-inicenter[1])^2+(x2-inicenter[2])^2) - x0))))
  
  D = 0.9
  delta = 0.02
  gamma = sqrt((3 * delta) / (4 * D))
  x_a = sqrt((4 * D) / (3 * delta)) * acosh(sqrt(20)) - 5.0

  f = D + 40 * delta * sech(gamma * (x[1] - x_a))^2

  H = max(f, b + equations.threshold_limiter)

  v1 = 0.0
  v2 = 0.0

  # Idea: Velocity only on most parts of the wave, not everywhere (x_a => x coordinate of wave maximum)
  if 1 <= x[1] || x[1] <= 2 * x_a - 1
    #v1 = sqrt(equations.gravity/D) * H
  end

  return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_wave

function boundary_condition_subcritical_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                                surface_flux_function, equations::ShallowWaterEquations2D)
  # Impulse from inside, height and bottom from outside
  u_outer = SVector(equations.threshold_limiter, u_inner[2], u_inner[3], 0.0)

  # calculate the boundary flux
  flux = surface_flux_function(u_inner, u_outer, normal_direction, equations)
  
  return flux
end

dirichlet_bc = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict( :Bottom => boundary_condition_slip_wall,
                            :Top    => boundary_condition_slip_wall,
                            :Right  => dirichlet_bc,
                            :Left   => boundary_condition_slip_wall )

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

meshfile_city = joinpath(@__DIR__, "seaside.mesh")
mesh = UnstructuredMesh2D(meshfile_city)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                   boundary_conditions=boundary_conditions)
                                   
###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 45.0)
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

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(equations.threshold_limiter,),
                                                     variables=(Trixi.waterheight,))

sol = solve(ode, SSPRK43(stage_limiter!),
            dt=1.0, save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
