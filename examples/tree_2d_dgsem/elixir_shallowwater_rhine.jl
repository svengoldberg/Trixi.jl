###################################################################################################
# Script used to simulate a dam break problem over a B-spline interpolated section of the         #
# Rhine river valley                                                                              #
###################################################################################################

using LinearAlgebra
using OrdinaryDiffEq
using Trixi

# Get root directory
dir_path = pkgdir(Trixi)#BottomTopography)

# Define data path
Rhine_data = string(dir_path, "/examples/TrixiBottomTopography/data/rhine_data_2d_20.txt")

# B-spline interpolation of the underlying data
spline_struct = BicubicBSpline(Rhine_data)
spline_func(x,y) = spline_interpolation(spline_struct, x, y)

equations = ShallowWaterEquations2D(gravity_constant=9.81, H0=100.0)

cfl = 0.1

#=
  New init:
    periodic: LGL=3, ref=4, T=10. => Good results!
      (Cylinder simulation works with those parameters as well)
  New init (cylinder):
    periodic: LGL=4, ref=3, aplha_max=0.6, T=20. => Good results!
              LGL=4, ref=5, aplha_max=0.6, T=20. => Good results! (RUNTIME!)
=#

#=
  Do some testing how the impact of the threshold in indicators_2d.jl is 
  old value: 1e-8 produced more 'crashes' [delT < 1e-6]) than large value (this run tested with 1e-4)
    => New th_indicator: Cylinder init runs with LGL=RefLvl=4, alpha_max=.6, T=20
  Dirichlet works with some parameters (use lower T (maybe 8)-> runtime! As water flows out of domain, not much happens)
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
#boundary_condition = BoundaryConditionDirichlet(initial_condition) #boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)

surface_flux = (FluxHydrostaticReconstruction(flux_hll_cn, hydrostatic_reconstruction_chen_noelle),
                flux_nonconservative_chen_noelle)
basis = LobattoLegendreBasis(4)

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
                initial_refinement_level=3,
                n_cells_max=10_000
                #,periodicity=false
               )

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver
                                   #;boundary_conditions=boundary_condition)
                                    )
###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
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

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(equations.threshold_limiter,),
                                                     variables=(Trixi.waterheight,))

sol = solve(ode, SSPRK43(stage_limiter!),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks, adaptive=false);
summary_callback() # print the timer summary
