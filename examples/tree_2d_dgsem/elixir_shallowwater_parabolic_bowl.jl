
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations2D(gravity_constant=9.81)
cfl = 0.1

#= 
  IDEA TO GET AROUND NOT ACHIEVING ref lvl 7: use AMR; current problem: It looks like that AMR destroys bottom topography
    and with that also the water height. Even after one time step the bottom looks different
    When this idea should be followed, I have to change the interpolation of b in the coarsening/refining to set to initial value at x
    Problem: Getting the initial condition within there, requires a lot of adjustment to existing code and adding new parameter
    only for that special case.
  Current ref methoded used: "Inverse IndicatorHennemannGassner", that means indicator giving 0 returns 1 and the other way round
  Why? To guarantee that there are not too much cells
    => Only the important (wet) cells are refined and the other ones are coarsend or nothing happens
=#

# Best run until now: CFL=0.1, LGL=6, RefLvl=6, alpha_max=0.6 (around 25 minutes)

# Implemented based on Wintermeyer (scaled on half of domain)
function parabolic_bowl_analytic_2D_H(gravity, x,t)
  a = 1
  h_0 = 0.1
  σ = 0.5
  ω = sqrt(2*gravity*h_0)/a

  H = 0.5 * (σ * h_0/a^2 * (2*2*x[1]*cos(ω*t) + 2*2*x[2]*sin(ω*t)- σ) + h_0)
  return H
end

# Implemented based on Wintermeyer (scaled on half of domain)
function initial_condition_parabolic_bowl(x, t, equations:: ShallowWaterEquations2D)
  a = 1
  h_0 = 0.1
  σ = 0.5
  ω = sqrt(2*equations.gravity*h_0)/a

  v1 = -σ*ω*sin(ω*t)
  v2 = -σ*ω*cos(ω*t)

  b = 0.5 * (h_0 * ((2*x[1])^2 + (2*x[2])^2)/a^2)

  H = max(b, parabolic_bowl_analytic_2D_H(equations.gravity, x, t))

  H = max(H, b + equations.threshold_limiter)
  return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_parabolic_bowl
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
# Create the TreeMesh for the domain [-1, 1]^2

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000
                )

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                    extra_analysis_integrals=(energy_kinetic,
                                                              energy_internal,
                                                              lake_at_rest_error))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true)

# Enabling AMR
# Look out for "bug" within AMR refinement to not change bottom topography (use init cond. or smthg)
# Now test how it looks when only wet part is refined (outer cells are at med_level from start on)

# Do test runs to guarantee that e.g Kelvin-Helmholtz AMR elixir gives same solution and works with modified AMR
#=
amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=.5,
                                          alpha_min=0.001,
                                          alpha_smooth=true,
                                          variable=waterheight_pressure)                                          
amr_controller = ControllerThreeLevel(semi, amr_indicator;
                                      base_level=5,
                                      med_level=5, med_threshold=0.1,
                                      max_level=6, max_threshold=0.5)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=1,
                           adapt_initial_condition=true)                                      
=#

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(equations.threshold_limiter,),
                                                     variables=(Trixi.waterheight,))

###############################################################################
# run the simulation

stepsize_callback = StepsizeCallback(cfl=cfl)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,stepsize_callback)
#callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,stepsize_callback, amr_callback)

sol = solve(ode, SSPRK43(stage_limiter!), dt=1.0,
            save_everystep=false, callback=callbacks, adaptive=false);
 
summary_callback() # print the timer summary
