
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations1D(gravity_constant=9.812, H0=1.75)

function initial_condition_stone_throw(x, t, equations::ShallowWaterEquations1D)
  # Set up polar coordinates
  inicenter = 0.15
  x_norm = x[1] - inicenter[1]
  r = abs(x_norm)

  # Calculate primitive variables
  H = equations.H0
  # v = 0.0 # for well-balanced test
  v = r < 0.6 ? 1.75 : 0.0 # for stone throw

  b = (  1.5 / exp( 0.5 * ((x[1] - 1.0)^2 ) )
     + 0.75 / exp( 0.5 * ((x[1] + 1.0)^2 ) ) )

  return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_stone_throw

boundary_condition = boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (flux_lax_friedrichs, flux_nonconservative_fjordholm_etal)

basis = LobattoLegendreBasis(4)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=waterheight_pressure)

volume_integral=VolumeIntegralShockCapturingHG(indicator_sc;
                                               volume_flux_dg=volume_flux,
                                               volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Create the TreeMesh for the domain [-3, 3]

coordinates_min = -3.0
coordinates_max = 3.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000,
                periodicity=false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     save_analysis=false,
                                     extra_analysis_integrals=(lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)                                     

save_solution = SaveSolutionCallback(interval=analysis_interval,
																     save_initial_solution=true,
																     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

sol = solve(ode, SSPRK43(), abstol=1.0e-7, reltol=1.0e-7,
					  save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary