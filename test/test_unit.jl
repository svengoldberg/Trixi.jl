module TestUnit

using Test
using SimpleMock
using Documenter
using Trixi

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# Run various unit (= non-elixir-triggered) tests
@testset "Unit tests" begin
  @testset "SerialTree" begin
    @testset "constructors" begin
      @test_nowarn Trixi.SerialTree(Val(1), 10, 0.0, 1.0)
    end

    @testset "helper functions" begin
      t = Trixi.SerialTree(Val(1), 10, 0.0, 1.0)
      @test_nowarn display(t)
      @test Trixi.ndims(t) == 1
      @test Trixi.has_any_neighbor(t, 1, 1) == true
      @test Trixi.isperiodic(t, 1) == true
      @test Trixi.n_children_per_cell(t) == 2
      @test Trixi.n_children_per_cell(2) == 4
      @test Trixi.n_directions(t) == 2
    end

    @testset "refine!/coarsen!" begin
      t = Trixi.SerialTree(Val(1), 10, 0.0, 1.0)
      @test Trixi.refine!(t) == [1]
      @test Trixi.coarsen!(t) == [1]
      @test Trixi.refine!(t) == [1]
      @test Trixi.coarsen!(t, 1) == [1]
      @test Trixi.coarsen!(t) == Int[] # Coarsen twice to check degenerate case of single-cell tree
      @test Trixi.refine!(t) == [1]
      @test Trixi.refine!(t) == [2,3]
      @test Trixi.coarsen_box!(t, [-0.5], [0.0]) == [2]
      @test Trixi.coarsen_box!(t, 0.0, 0.5) == [3]
      @test isnothing(Trixi.reset_data_structures!(t))
    end
  end

  @testset "ParallelTree" begin
    @testset "constructors" begin
      @test_nowarn Trixi.ParallelTree(Val(1), 10, 0.0, 1.0)
    end

    @testset "helper functions" begin
      t = Trixi.ParallelTree(Val(1), 10, 0.0, 1.0)
      @test isnothing(display(t))
      @test isnothing(Trixi.reset_data_structures!(t))
    end
  end

  @testset "TreeMesh" begin
    @testset "constructors" begin
      @test TreeMesh{1, Trixi.SerialTree{1}}(1, 5.0, 2.0) isa TreeMesh
    end
  end

  @testset "ParallelTreeMesh" begin
    @testset "partition!" begin
      @testset "mpi_nranks() = 2" begin
        mock((Trixi.mpi_nranks) => () -> 2) do _
          mesh = TreeMesh{2, Trixi.ParallelTree{2}}(30, (0.0, 0.0), 1)
          # Refine twice
          Trixi.refine!(mesh.tree)
          Trixi.refine!(mesh.tree)

          # allow_coarsening = true
          Trixi.partition!(mesh)
          # Use parent for OffsetArray
          @test parent(mesh.n_cells_by_rank) == [11, 10]
          @test mesh.tree.mpi_ranks[1:21] ==
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          @test parent(mesh.first_cell_by_rank) == [1, 12]

          # allow_coarsening = false
          Trixi.partition!(mesh; allow_coarsening=false)
          @test parent(mesh.n_cells_by_rank) == [11, 10]
          @test mesh.tree.mpi_ranks[1:21] ==
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          @test parent(mesh.first_cell_by_rank) == [1, 12]
        end
      end

      @testset "mpi_nranks() = 3" begin
        mock((Trixi.mpi_nranks) => () -> 3) do _
          mesh = TreeMesh{2, Trixi.ParallelTree{2}}(100, (0.0, 0.0), 1)
          # Refine twice
          Trixi.refine!(mesh.tree)
          Trixi.refine!(mesh.tree)

          # allow_coarsening = true
          Trixi.partition!(mesh)
          # Use parent for OffsetArray
          @test parent(mesh.n_cells_by_rank) == [11, 5, 5]
          @test mesh.tree.mpi_ranks[1:21] ==
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
          @test parent(mesh.first_cell_by_rank) == [1, 12, 17]

          # allow_coarsening = false
          Trixi.partition!(mesh; allow_coarsening=false)
          @test parent(mesh.n_cells_by_rank) == [9, 6, 6]
          @test mesh.tree.mpi_ranks[1:21] ==
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
          @test parent(mesh.first_cell_by_rank) == [1, 10, 16]
        end
      end

      @testset "mpi_nranks() = 9" begin
        mock((Trixi.mpi_nranks) => () -> 9) do _
          mesh = TreeMesh{2, Trixi.ParallelTree{2}}(1000, (0.0, 0.0), 1)
          # Refine twice
          Trixi.refine!(mesh.tree)
          Trixi.refine!(mesh.tree)
          Trixi.refine!(mesh.tree)
          Trixi.refine!(mesh.tree)

          # allow_coarsening = true
          Trixi.partition!(mesh)
          # Use parent for OffsetArray
          @test parent(mesh.n_cells_by_rank) == [44, 37, 38, 37, 37, 37, 38, 37, 36]
          @test parent(mesh.first_cell_by_rank) == [1, 45, 82, 120, 157, 194, 231, 269, 306]
        end
      end

      @testset "mpi_nranks() = 3 non-uniform" begin
        mock((Trixi.mpi_nranks) => () -> 3) do _
          mesh = TreeMesh{2, Trixi.ParallelTree{2}}(100, (0.0, 0.0), 1)
          # Refine whole tree
          Trixi.refine!(mesh.tree)
          # Refine left leaf
          Trixi.refine!(mesh.tree, [2])

          # allow_coarsening = true
          Trixi.partition!(mesh)
          # Use parent for OffsetArray
          @test parent(mesh.n_cells_by_rank) == [6, 1, 2]
          @test mesh.tree.mpi_ranks[1:9] == [0, 0, 0, 0, 0, 0, 1, 2, 2]
          @test parent(mesh.first_cell_by_rank) == [1, 7, 8]

          # allow_coarsening = false
          Trixi.partition!(mesh; allow_coarsening=false)
          @test parent(mesh.n_cells_by_rank) == [5, 2, 2]
          @test mesh.tree.mpi_ranks[1:9] == [0, 0, 0, 0, 0, 1, 1, 2, 2]
          @test parent(mesh.first_cell_by_rank) == [1, 6, 8]
        end
      end

      @testset "not enough ranks" begin
        mock((Trixi.mpi_nranks) => () -> 3) do _
          mesh = TreeMesh{2, Trixi.ParallelTree{2}}(100, (0.0, 0.0), 1)

          # Only one leaf
          @test_throws AssertionError(
            "Too many ranks to properly partition the mesh!") Trixi.partition!(mesh)

          # Refine to 4 leaves
          Trixi.refine!(mesh.tree)

          # All four leaves will need to be on one rank to allow coarsening
          @test_throws AssertionError(
            "Too many ranks to properly partition the mesh!") Trixi.partition!(mesh)
          @test_nowarn Trixi.partition!(mesh; allow_coarsening=false)
        end
      end
    end
  end

  @testset "interpolation" begin
    @testset "nodes and weights" begin
      @test Trixi.gauss_nodes_weights(1) == ([0.0], [2.0])
    end

    @testset "multiply_dimensionwise" begin
      nodes_in  = [0.0, 0.5, 1.0]
      nodes_out = [0.0, 1/3, 2/3, 1.0]
      matrix = Trixi.polynomial_interpolation_matrix(nodes_in, nodes_out)
      data_in = [3.0 4.5 6.0]
      @test isapprox(Trixi.multiply_dimensionwise(matrix, data_in), [3.0 4.0 5.0 6.0])

      n_vars   = 3
      size_in  = 4
      size_out = 5
      matrix   = randn(size_out, size_in)
      # 1D
      data_in  = randn(n_vars, size_in)
      data_out = Trixi.multiply_dimensionwise_naive(matrix, data_in)
      @test isapprox(data_out, Trixi.multiply_dimensionwise(matrix, data_in))
      # 2D
      data_in  = randn(n_vars, size_in, size_in)
      data_out = Trixi.multiply_dimensionwise_naive(matrix, data_in)
      @test isapprox(data_out, Trixi.multiply_dimensionwise(matrix, data_in))
      # 3D
      data_in  = randn(n_vars, size_in, size_in, size_in)
      data_out = Trixi.multiply_dimensionwise_naive(matrix, data_in)
      @test isapprox(data_out, Trixi.multiply_dimensionwise(matrix, data_in))
    end
  end

  @testset "L2 projection" begin
    @testset "calc_reverse_upper for LGL" begin
      @test isapprox(Trixi.calc_reverse_upper(2, Val(:gauss_lobatto)), [[0.25, 0.25] [0.0, 0.5]])
    end
    @testset "calc_reverse_lower for LGL" begin
      @test isapprox(Trixi.calc_reverse_lower(2, Val(:gauss_lobatto)), [[0.5, 0.0] [0.25, 0.25]])
    end
  end

  @testset "containers" begin
    # Set up mock container
    mutable struct MyContainer <: Trixi.AbstractContainer
      data::Vector{Int}
      capacity::Int
      length::Int
      dummy::Int
    end
    function MyContainer(data, capacity)
      c = MyContainer(Vector{Int}(undef, capacity+1), capacity, length(data), capacity+1)
      c.data[1:length(data)] .= data
      return c
    end
    MyContainer(data::AbstractArray) = MyContainer(data, length(data))
    Trixi.invalidate!(c::MyContainer, first, last) = (c.data[first:last] .= 0; c)
    function Trixi.raw_copy!(target::MyContainer, source::MyContainer, first, last, destination)
      Trixi.copy_data!(target.data, source.data, first, last, destination)
      return target
    end
    Trixi.move_connectivity!(c::MyContainer, first, last, destination) = c
    Trixi.delete_connectivity!(c::MyContainer, first, last) = c
    Trixi.reset_data_structures!(c::MyContainer) = (c.data = Vector{Int}(undef, c.capacity+1); c)
    function Base.:(==)(c1::MyContainer, c2::MyContainer)
      return (c1.capacity == c2.capacity &&
              c1.length == c2.length &&
              c1.dummy == c2.dummy &&
              c1.data[1:c1.length] == c2.data[1:c2.length])
    end

    @testset "size" begin
      c = MyContainer([1, 2, 3])
      @test size(c) == (3,)
    end

    @testset "resize!" begin
      c = MyContainer([1, 2, 3])
      @test length(resize!(c, 2)) == 2
    end

    @testset "copy!" begin
      c1 = MyContainer([1, 2, 3])
      c2 = MyContainer([4, 5])
      @test Trixi.copy!(c1, c2, 2, 1, 2) == MyContainer([1, 2, 3]) # no-op

      c1 = MyContainer([1, 2, 3])
      c2 = MyContainer([4, 5])
      @test Trixi.copy!(c1, c2, 1, 2, 2) == MyContainer([1, 4, 5])

      c1 = MyContainer([1, 2, 3])
      @test Trixi.copy!(c1, c2, 1, 2) == MyContainer([1, 4, 3])

      c1 = MyContainer([1, 2, 3])
      @test Trixi.copy!(c1, 2, 3, 1) == MyContainer([2, 3, 3])

      c1 = MyContainer([1, 2, 3])
      @test Trixi.copy!(c1, 1, 3) == MyContainer([1, 2, 1])
    end

    @testset "move!" begin
      c = MyContainer([1, 2, 3])
      @test Trixi.move!(c, 1, 1) == MyContainer([1, 2, 3]) # no-op

      c = MyContainer([1, 2, 3])
      @test Trixi.move!(c, 1, 2) == MyContainer([0, 1, 3])
    end

    @testset "swap!" begin
      c = MyContainer([1,2])
      @test Trixi.swap!(c, 1, 1) == MyContainer([1, 2]) # no-op

      c = MyContainer([1,2])
      @test Trixi.swap!(c, 1, 2) == MyContainer([2,1])
    end

    @testset "erase!" begin
      c = MyContainer([1, 2])
      @test Trixi.erase!(c, 2, 1) == MyContainer([1, 2]) # no-op

      c = MyContainer([1, 2])
      @test Trixi.erase!(c, 1) == MyContainer([0, 2])
    end

    @testset "remove_shift!" begin
      c = MyContainer([1, 2, 3, 4])
      @test Trixi.remove_shift!(c, 2, 1) == MyContainer([1, 2, 3, 4]) # no-op

      c = MyContainer([1, 2, 3, 4])
      @test Trixi.remove_shift!(c, 2, 2) == MyContainer([1, 3, 4], 4)

      c = MyContainer([1, 2, 3, 4])
      @test Trixi.remove_shift!(c, 2) == MyContainer([1, 3, 4], 4)
    end

    @testset "remove_fill!" begin
      c = MyContainer([1, 2, 3, 4])
      @test Trixi.remove_fill!(c, 2, 1) == MyContainer([1, 2, 3, 4]) # no-op

      c = MyContainer([1, 2, 3, 4])
      @test Trixi.remove_fill!(c, 2, 2) == MyContainer([1, 4, 3], 4)
    end

    @testset "reset!" begin
      c = MyContainer([1, 2, 3])
      @test Trixi.reset!(c, 2) == MyContainer(Int[], 2)
    end
  end

  @testset "example elixirs" begin
    @test basename(examples_dir()) == "examples"
    @test !isempty(get_examples())
    @test endswith(default_example(), "elixir_advection_basic.jl")
  end

  @testset "DG L2 mortar container debug output" begin
    c2d = Trixi.L2MortarContainer2D{Float64, 1, 1}(1)
    @test isnothing(display(c2d))
    c3d = Trixi.L2MortarContainer3D{Float64, 1, 1}(1)
    @test isnothing(display(c3d))
  end

  @testset "Printing indicators/controllers" begin
    # OBS! Constructing indicators/controllers using the parameters below doesn't make sense. It's
    # just useful to run basic tests of `show` methods.

    c = ControllerThreeLevelCombined(1, 2, 3, 10.0, 11.0, 12.0, "primary", "secondary", "cache")
    @test_nowarn show(stdout, c)

    indicator_hg = IndicatorHennemannGassner(1.0, 0.0, true, "variable", "cache")
    @test_nowarn show(stdout, indicator_hg)

    indicator_loehner = IndicatorLöhner(1.0, "variable", (; cache=nothing))
    @test_nowarn show(stdout, indicator_loehner)

    indicator_max = IndicatorMax("variable", (; cache=nothing))
    @test_nowarn show(stdout, indicator_max)
  end

  @testset "LBM 2D constructor" begin
    # Neither Mach number nor velocity set
    @test_throws ErrorException LatticeBoltzmannEquations2D(Ma=nothing, Re=1000)
    # Both Mach number and velocity set
    @test_throws ErrorException LatticeBoltzmannEquations2D(Ma=0.1, Re=1000, u0=1)
    # Neither Reynolds number nor viscosity set
    @test_throws ErrorException LatticeBoltzmannEquations2D(Ma=0.1, Re=nothing)
    # Both Reynolds number and viscosity set
    @test_throws ErrorException LatticeBoltzmannEquations2D(Ma=0.1, Re=1000, nu=1)

    # No non-dimensional values set
    @test LatticeBoltzmannEquations2D(Ma=nothing, Re=nothing, u0=1, nu=1) isa LatticeBoltzmannEquations2D
  end

  @testset "LBM 3D constructor" begin
    # Neither Mach number nor velocity set
    @test_throws ErrorException LatticeBoltzmannEquations3D(Ma=nothing, Re=1000)
    # Both Mach number and velocity set
    @test_throws ErrorException LatticeBoltzmannEquations3D(Ma=0.1, Re=1000, u0=1)
    # Neither Reynolds number nor viscosity set
    @test_throws ErrorException LatticeBoltzmannEquations3D(Ma=0.1, Re=nothing)
    # Both Reynolds number and viscosity set
    @test_throws ErrorException LatticeBoltzmannEquations3D(Ma=0.1, Re=1000, nu=1)

    # No non-dimensional values set
    @test LatticeBoltzmannEquations3D(Ma=nothing, Re=nothing, u0=1, nu=1) isa LatticeBoltzmannEquations3D
  end

  @testset "LBM 2D functions" begin
    # Set up LBM struct and dummy distribution
    equation = LatticeBoltzmannEquations2D(Ma=0.1, Re=1000)
    u = Trixi.equilibrium_distribution(1, 2, 3, equation)

    # Component-wise velocity
    @test isapprox(Trixi.velocity(u, 1, equation), 2)
    @test isapprox(Trixi.velocity(u, 2, equation), 3)
  end

  @testset "LBM 3D functions" begin
    # Set up LBM struct and dummy distribution
    equation = LatticeBoltzmannEquations3D(Ma=0.1, Re=1000)
    u = Trixi.equilibrium_distribution(1, 2, 3, 4, equation)

    # Component-wise velocity
    @test isapprox(velocity(u, 1, equation), 2)
    @test isapprox(velocity(u, 2, equation), 3)
    @test isapprox(velocity(u, 3, equation), 4)
  end

  @testset "LBMCollisionCallback" begin
    # Printing of LBM collision callback
    callback = LBMCollisionCallback()
    @test_nowarn show(stdout, callback)
    println()
    @test_nowarn show(stdout, "text/plain", callback)
    println()
  end

  # Test docstrings
  DocMeta.setdocmeta!(Trixi, :DocTestSetup, :(using Trixi); recursive=true)
  doctest(Trixi, manual=false)
end

end #module