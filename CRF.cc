/*
 * This code intends to solve a simplified model of chemically reacting flows.
 * This involves tracking the velocity, pressure, and temperature of a gas.
 * We also need to keep track of the concentrations of the chemical species
 * within the gas.
 *
 * This leads to Stokes equations describing the velocity and pressure,
 * and advection equations describing the temperature and concentrations.
 *
 * Our PDEs in their strong form are
 * -2 \mu div( \epsilon(u) ) + grad( p ) = f_a  (1)
 * - div( u )                            = f_b  (2)
 * \beta \cdot grad( T )                 = f_T  (3)
 * \beta \cdot grad( s_1 )               = f_1  (4)
 * ...
 * \beta \cdot grad( s_m )               = f_m  (5)
 *
 * where we need to solve for:
 * u = velocities (vector) (e.g. (u_x, u_y, u_z) for 3D)
 * p = pressure (scalar)
 * T = temperature (scalar)
 * s_1 = concentration of chemical species 1 (scalar)
 * s_m = concentration of chemical species m (scalar; m total chemical species)
 *
 * and we know:
 * \mu = viscosity of the gas
 * \beta = wind field acting on the gas -- this should be u, but we define a known wind field as a simplification for the moment.
 * \epsilon = symmetric gradient -- i.e. 1/2(grad( u ) + [grad( u )]^T)
 * along with associated boundary conditions which will be discussed later.
 *
 * We convert to the weak form by multiplying be test functions and integrating over the domain. For the advection
 * equations, we use streamline diffusion test functions. Integration by parts is used on (1).
 *
 * The resulting matrix equations are:
 * | A  B^T 0   0   ... 0  | | U  |  | F_a - G_{up}|
 * | B  0   0   0   ... 0  | | P  |  | F_b         |
 * | 0  0   C_T 0   ... 0  | | T  | =| F_T - G_T   |
 * | 0  0   0   C_1 ... 0  | | S_1|  | F_1 - G_1   |
 * | ...................   | | ...|  | ...         |
 * | 0  0   0   0   ... C_m| | S_m|  | F_m - G_m   |
 *
 * where:
 * A_{ij}   = ( \epsilon(\phi_{u,i}), 2 \mu \epsilon(\phi_{u,j}) )_\Omega
 * B_{ij}   = (\phi_{p,i}, - div(\phi_{u,j}) )_\Omega
 * C_{X,ij} = (\phi_{X,i}+\delta\beta\cdot grad(\phi_{X,i}), \beta\cdot grad( \phi_{X,j} )_\Omega - (\beta\cdot n \phi_{X,i}, \phi_{X,j} )_{\partial\Omega-}
 *
 * F_{a,i} = ( \phi_{u,i}, f_a )_\Omega
 * F_{b,i} = ( \phi_{p,i}, f_b )_\Omega
 * F_{X,i} = ( \phi_{X,i} + \delta\beta\cdot grad( \phi_{X,i} ), f_{X,i} )_\Omega
 *
 * G_{up,i} = ( \phi_u, g_n )_{\Gamma_N} --> g_n = Neumann BC for stokes --> n\cdot[pI-2\mu\epsilon(u)] on \Gamma_N
 * G_{X,i}  = (\beta\cdot n \phi_{X,i}, g_X)_{\partial\Omega inflow) --> X = g_X on \partial\Omega inflow
 */



/*
 * HEADER FILES
 */

// include necessary deal.II header files
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

// include standard library header files
#include <iostream>
#include <fstream>
//#include <memory>


namespace CRF {
  using namespace dealii;

  /*
   * 1. BoundaryValues
   */

  // Declaration

  template<int dim>
  class BoundaryValues : public Function<dim> {
  public:
    explicit BoundaryValues(const unsigned int n_rxns)
        : Function<dim>(dim + 1 + 1 + n_rxns) {}

    double value(const Point<dim> &p,
                 unsigned int component) const override;

    void vector_value(const Point<dim> &p,
                      Vector<double> &values) const override;
  };



  // Implementation

  template<int dim>
  double BoundaryValues<dim>::value(const Point<dim> &p,
                                    const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));

    // velocities away from origin
    if (component < dim)
    {
      return (p[component] < 0 ? -1 : (p[component] > 0 ? 1 : 0));
    }
    else if (component == dim)
      return 0.; // zero BC for pressure
    else if (component == dim + 1) // temperature
    {
      if (p[1] == -1.0) // prescribe BV at bottom of domain
        return 293.0; // Room temperature in Kelvin
      else
        return 0.0;
    }
    else if (component == dim + 2 || component == dim + 3) // CH2O or OH
    {
      if (p[1] == -1.0)
        return 1.0e-9;
      else
        return 0.;
    }
    else // HCO or H2O
      return 0.;
  }


  template<int dim>
  void BoundaryValues<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values[c] = BoundaryValues<dim>::value(p, c);
  }



  /*
   * 2. CRFProblem
   */
  // Declaration

  template<int dim>
  class CRFProblem {
  public:
    CRFProblem(unsigned int stokes_degree,
               unsigned int advect_degree,
               unsigned int n_rxns,
               double viscosity);

    void run(unsigned int refinements);

  private:
    void setup_dofs();

    void assemble_system();

    void solve();

    void refine_mesh();

    void output_results(unsigned int cycle) const;

    const unsigned int stokes_degree;
    const unsigned int advect_degree;
    const unsigned int n_rxns;
    const double viscosity;

    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;

    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockSparsityPattern preconditioner_sparsity_pattern;
    BlockSparseMatrix<double> preconditioner_matrix;

    BlockVector<double> solution;
    BlockVector<double> previous_solution;
    BlockVector<double> system_rhs;
  };



  // Implementation

  template<int dim>
  CRFProblem<dim>::CRFProblem(const unsigned int stokes_degree,
                              const unsigned int advect_degree,
                              const unsigned int n_rxns,
                              const double viscosity)
      : stokes_degree(stokes_degree), advect_degree(advect_degree), n_rxns(n_rxns), viscosity(viscosity),
        triangulation(Triangulation<dim>::maximum_smoothing), fe(FE_Q<dim>(stokes_degree + 1) ^ dim,    // velocities
                                                                 FE_Q<dim>(stokes_degree),          // pressure
                                                                 FE_Q<dim>(advect_degree),          // temperature
                                                                 FE_Q<dim>(advect_degree) ^n_rxns)   // chemical concentrations
      , dof_handler(triangulation)
  {}


  template<int dim>
  void CRFProblem<dim>::run(const unsigned int refinements) {
    // Attach a grid to the triangulation
    {
      std::vector<unsigned int> subdivisions(dim, 1);
      subdivisions[0] = 4;

      const Point<dim> bottom_left = (dim == 2 ?
                                      Point<dim>(-2, -1) :        // 2d case
                                      Point<dim>(-2, 0, -1));     // 3d case

      const Point<dim> top_right = (dim == 2 ?
                                    Point<dim>(2, 0) :          // 2d case
                                    Point<dim>(2, 1, 0));       // 3d case

      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                subdivisions,
                                                bottom_left,
                                                top_right);
    }

    // Set Dirichlet boundary
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->center()[dim - 1] == 0)
          face->set_all_boundary_ids(1);

    // Refine globally prior to solving the problem
    triangulation.refine_global(3 - dim);

    // solve over a few successively refined grids
    for (unsigned int cycle = 0; cycle < refinements; ++cycle) {
      std::cout << "Refinement cycle " << cycle << std::endl;

      if (cycle > 0)
        refine_mesh();

      // Solve the problem
      setup_dofs();

      // Non-linear loop
      double residual = 1.;
      unsigned int iteration_num = 0;
      while (residual > 1e-3)
      {
        std::cout << "Fixed point iteration " << iteration_num << std::endl << std::flush;
        std::cout << "Assembling..." << std::endl << std::flush;
        assemble_system();

        std::cout << "Solving..." << std::endl << std::flush;
        solve();
        std::cout << std::endl;

        BlockVector<double> diff = solution - previous_solution;
        residual = 0.;
        for (unsigned int i=0; i < dim + n_rxns + 1; ++i)
        {
          residual += diff.block(i).l2_norm()/previous_solution.block(i).l2_norm();
        }

        std::cout << "The residual between the previous and current solution is " << residual << std::endl << std::flush;

        previous_solution = solution;
         iteration_num += 1;
      }
      output_results(cycle);

      std::cout << std::endl;
    }
  }


  template<int dim>
  void CRFProblem<dim>::setup_dofs() {
    system_matrix.clear();
    preconditioner_matrix.clear();

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);

    // Renumber the DoF's so that solution variables are ordered in blocks

    std::vector<unsigned int> block_component(dim + 1 + 1 + n_rxns, 0); // Velocity block labelled 0
    block_component[dim] = 1;                                                    // Pressure block label
    block_component[dim + 1] = 2;                                                // Temperature block label

    // Chemical species blocks labels
    for (unsigned int chem = 0; chem < n_rxns; ++chem) {
      block_component[dim + 2 + chem] = 3 + chem;
    }

    DoFRenumbering::component_wise(dof_handler, block_component);

    // Set up hanging node and Dirichlet BC constraints
    {
      constraints.clear();

      FEValuesExtractors::Vector velocities(0);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               BoundaryValues<dim>(n_rxns),
                                               constraints,
                                               fe.component_mask(velocities));
    }

    constraints.close();

    const std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << dofs_per_block[0];
    for (unsigned int block = 1; block < dofs_per_block.size(); ++block) {
      std::cout << '+' << dofs_per_block[block];
    }
    std::cout << ')' << std::endl;

    {
      BlockDynamicSparsityPattern dsp(dofs_per_block.size(), dofs_per_block.size());

      for (unsigned int row = 0; row < dofs_per_block.size(); ++row)
        for (unsigned int col = 0; col < dofs_per_block.size(); ++col) {
          dsp.block(row, col).reinit(dofs_per_block[row], dofs_per_block[col]);
        }

      dsp.collect_sizes();

      DoFTools::make_sparsity_pattern(dof_handler,
                                      dsp,
                                      constraints,
                                      false);

      sparsity_pattern.copy_from(dsp);
    }

    {
      BlockDynamicSparsityPattern preconditioner_dsp(dofs_per_block.size(), dofs_per_block.size());

      for (unsigned int row = 0; row < dofs_per_block.size(); ++row)
        for (unsigned int col = 0; col < dofs_per_block.size(); ++col) {
          preconditioner_dsp.block(row, col).reinit(dofs_per_block[row], dofs_per_block[col]);
        }

      preconditioner_dsp.collect_sizes();

      DoFTools::make_sparsity_pattern(dof_handler,
                                      preconditioner_dsp,
                                      constraints,
                                      false);

      preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
    }

    system_matrix.reinit(sparsity_pattern);
    preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

    solution.reinit(dofs_per_block.size());
    previous_solution.reinit(dofs_per_block.size());
    system_rhs.reinit(dofs_per_block.size());

    for (unsigned int block = 0; block < dofs_per_block.size(); ++block) {
      solution.block(block).reinit(dofs_per_block[block]);
      previous_solution.block(block).reinit(dofs_per_block[block]);
      system_rhs.block(block).reinit(dofs_per_block[block]);
    }

    solution.collect_sizes();
    previous_solution.collect_sizes();
    system_rhs.collect_sizes();

    // Before velocities are calculated, start with them at 1 in each direction.
    previous_solution = 1;
    // Temperature of 1 Kelvin is physically unreasonable, so start it at room temperature
    previous_solution.block(2) *= 293.;
    // Set initial concentrations for CH2O and OH to be 0
    previous_solution.block(3) = 0;
    previous_solution.block(4) = 0;
  }


  template<int dim>
  void CRFProblem<dim>::assemble_system() {
    system_matrix = 0;
    system_rhs = 0;
    preconditioner_matrix = 0;

    // FIXME: why degree+2?
    QGauss<dim> quadrature_formula(stokes_degree + 2);

    // Values needed inside domain:
    // velocities -- symmetric gradient, value
    // pressure -- value
    // temperature -- values, gradients
    // chemical species -- values, gradients
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);

    // Boundary contributions from advection equations
    // (\beta\cdot n \phi_X, \phi_X)
    // n = update_normal_vectors
    // \phi_X = update_values
    FEFaceValues<dim> fe_face_values(fe,
                                     QGauss<dim - 1>(advect_degree + 1),
                                     update_values | update_quadrature_points |
                                     update_JxW_values | update_normal_vectors);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_preconditioner_matrix(dofs_per_cell, dofs_per_cell);

    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const BoundaryValues<dim> boundary_value(n_rxns);

    std::vector<Tensor<1, dim>> advection_direction(n_q_points);
    std::vector<double> temp_vals(n_q_points);
    std::vector<double> CH2O_values(n_q_points);
    std::vector<double> OH_values(n_q_points);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    const FEValuesExtractors::Scalar temperature(dim + 1);
    // FIXME: Is this the proper way to do this?
    std::vector<FEValuesExtractors::Scalar> chemicals(n_rxns);
    for (unsigned int chem = 0; chem < n_rxns; ++chem) {
      const FEValuesExtractors::Scalar new_chemical(dim + 2 + chem);
      chemicals[chem] = new_chemical;
    }

    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<double> div_phi_u(dofs_per_cell);

    std::vector<double> phi_p(dofs_per_cell);

    std::vector<Tensor<1, dim>> grad_phi_T(dofs_per_cell);
    std::vector<double> phi_T(dofs_per_cell);

    std::vector<std::vector<Tensor<1, dim>>> grad_phi_chem(n_rxns);
    for (unsigned int chem = 0; chem < n_rxns; ++chem) {
      std::vector<Tensor<1, dim>> new_grad_phi(dofs_per_cell);
      grad_phi_chem[chem] = new_grad_phi;
    }

    std::vector<std::vector<double>> phi_chem(n_rxns);
    for (unsigned int chem = 0; chem < n_rxns; ++chem) {
      std::vector<double> new_phi(dofs_per_cell);
      phi_chem[chem] = new_phi;
    }

    // Loop over all cells
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      fe_values.reinit(cell);
      local_matrix = 0;
      local_preconditioner_matrix = 0;
      local_rhs = 0;

      // Calculate advection based on previous velocity solutions
      fe_values[velocities].get_function_values(previous_solution,
                                                advection_direction);

      // The only right hand side is from chemical reactions
      // We model the reaction CH2O + OH -> HCO + H2O
      // So the right hand sides are:
      // CH2O: - (CH2O concentration) * (OH concentration) * k
      // OH: - (CH2O concentration) * (OH concentration) * k
      // HCO: (CH2O concentration) * (OH concentration) * k
      // H2O: (CH2O concentration) * (OH concentration) * k
      //
      // k is the rate constant which determines the speed of the reaction.
      // We calculate this with the modified Arrhenius equation k = A T^n exp(-E_a/[R*T])
      // where
      // A = pre-exponential factor with units cm^3/(mol*s*K^n).
      // T^n = Temperature dependence of the pre-exponential factor with units K.
      //    *note* the units of A are a bit odd, but it's based off the term A*T^n having units of cm^3/(mol*s).
      // n = dimensionless constant representing the power of T the rate constant depends on.
      // E_a = activation energy for the reaction with units J/mol.
      // R = universal gas constant with units J/mol.
      //
      // For the reaction CH2O + OH -> HCO + H2O we have
      // k = 7.82e7 * T^(1.63) * exp( 531 / T )
      // source: https://web.stanford.edu/group/haiwanglab/FFCM1/pages/trialrates.html
      fe_values[temperature].get_function_values(previous_solution,
                                                 temp_vals);
      fe_values[chemicals[0]].get_function_values(previous_solution,
                                                  CH2O_values);
      fe_values[chemicals[1]].get_function_values(previous_solution,
                                                  OH_values);


      // Calculate delta for streamline diffusion based off cell diameter
      const double delta = 0.1 * cell->diameter();

      // loop over all quadrature points
      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          // We are populating a matrix with (phi_i, phi_j) where ( , ) describes
          // the complete bilinear form of our weak form.
          // Since we loop over i and then j, we can save computational energy by
          // precalculating all phi terms and then just call them within the loop.
          phi_u[k] = fe_values[velocities].value(k, q);
          symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient(k, q);
          div_phi_u[k] = fe_values[velocities].divergence(k, q);

          phi_p[k] = fe_values[pressure].value(k, q);

          grad_phi_T[k] = fe_values[temperature].gradient(k, q);
          phi_T[k] = fe_values[temperature].value(k, q);

          // FIXME: Is this the way to do things?
          for (unsigned int chem = 0; chem < n_rxns; ++chem) {
            phi_chem[chem][k] = fe_values[chemicals[chem]].value(k, q);
            grad_phi_chem[chem][k] = fe_values[chemicals[chem]].gradient(k, q);
          }
        }

        // Value of JxW for quadrature point
        const double dx = fe_values.JxW(q);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            // contributions from stokes equations and temperature advection equation
            local_matrix(i, j) +=
                (
                    2 * viscosity * (symgrad_phi_u[i] * symgrad_phi_u[j])                   // stokes
                    - div_phi_u[i] * phi_p[j]                                               // stokes
                    - phi_p[i] * div_phi_u[j]                                               // stokes
                    + (phi_T[i] + delta * (advection_direction[q] * grad_phi_T[i])) *       // temperature
                      (advection_direction[q] * grad_phi_T[j])                              // temperature
                ) * dx;

            // Contributions to mass matrix preconditioner. We will only use the pressure mass matrix.
            local_preconditioner_matrix(i, j) += (phi_p[i] * phi_p[j]) * dx;

            // contributions from chemical concentrations advection equations
            for (unsigned int chem = 0; chem < n_rxns; ++chem) {
              local_matrix(i, j) +=
                  (
                      (phi_chem[chem][i] + delta * (advection_direction[q] * grad_phi_chem[chem][i]))
                      * (advection_direction[q] * grad_phi_chem[chem][j])
                  ) * dx;
            }

          }


          // Shape functions are only non-zero in one component
          const unsigned int component_i = fe.system_to_component_index(i).first;
          // The right hand side is only non-zero for the chemical species, so only calculate then
          if (component_i > dim + 1)
          {
            // FIXME: a bit hard-coded
            const double direction = (component_i < dim + 4) ? -1. : 1.;
            const double rxn_rate = 7.82e7 * std::pow(std::abs(temp_vals[q]), 1.63) * std::exp(531. / temp_vals[q]);
            const double rhs_val = direction * rxn_rate * CH2O_values[q] * OH_values[q];
            if (isnan(rhs_val))
            {
              std::cout << "T^n = " << std::pow(temp_vals[q], 1.63)
                        << ", exp(531/T) = " << std::exp(531 / temp_vals[q])
                        << std::endl;
            }

            local_rhs(i) += ( fe_values.shape_value(i, q)
                            + delta * advection_direction[q] * fe_values.shape_grad(i, q) )
                            * rhs_val * dx;
          }
        }
      }

      // We now check to see if any face on the cell is on an inflow boundary.
      // If so, the local matrix and rhs get contributions from the advection equations.
      for (const auto &face : cell->face_iterators()) {
        if (face->at_boundary()) {
          fe_face_values.reinit(cell, face);

          std::vector<Tensor<1, dim>> advection_direction(n_face_q_points);
          fe_face_values[velocities].get_function_values(previous_solution,
                                                         advection_direction);

          // Loop over quadrature points and see if each point is on inflow or outflow direction.
          // The scalar product between the advection direction and the normal direction should
          // be negative on the inflow boundary. This is because the normal direction points outside
          // the domain while the advection direction points inside.
          for (unsigned int q = 0; q < n_face_q_points; ++q) {
            // define variables to make things easier to read
            const Point<dim> q_point = fe_face_values.quadrature_point(q);
            const Tensor<1, dim> normal_vector = fe_face_values.normal_vector(q);

            if (advection_direction[q] * normal_vector < 0.) {
              // If the face is part of the inflow boundary, compute the contributions to the
              // local matrix and right hand side.
              const double dx = fe_face_values.JxW(q);

              for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                phi_T[k] = fe_face_values[temperature].value(k, q);
                for (unsigned int chem = 0; chem < n_rxns; ++chem) {
                  phi_chem[chem][k] = fe_face_values[chemicals[chem]].value(k, q);
                }
              }

              for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                  double phi_times_phi = phi_T[i] * phi_T[j];
                  for (unsigned int chem = 0; chem < n_rxns; ++chem) {
                    phi_times_phi += phi_chem[chem][i] * phi_chem[chem][j];
                  }

                  local_matrix(i, j) -= (advection_direction[q] * normal_vector)
                                        * phi_times_phi * dx;
                }

                double phi_times_bv = phi_T[i] * boundary_value.value(q_point, dim + 1);
                for (unsigned int chem = 0; chem < n_rxns; ++chem) {
                  phi_times_bv += phi_chem[chem][i] * boundary_value.value(q_point, dim + 2 + chem);
                }

                local_rhs(i) -= (advection_direction[q] * normal_vector)
                                * phi_times_bv * dx;
              }
            }
          }

        }
      }

      // Now that all calculations for the cell are complete, update the global matrix
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(local_matrix,
                                             local_rhs,
                                             local_dof_indices,
                                             system_matrix,
                                             system_rhs);

      constraints.distribute_local_to_global(local_preconditioner_matrix,
                                             local_dof_indices,
                                             preconditioner_matrix);
    }
  }


  /* We have a system of the form
   *    | A B^T 0   0   ... 0   | | U   |   | F   |
   *    | B 0   0   0   ... 0   | | P   |   | G   |
   *    | 0 0   C_T 0   ... 0   | | T   | = | F_T |
   *    | 0 0   0   C_1 ... 0   | | S_1 |   | F_1 |
   *    |       ...             | | ... |   | ... |
   *    | 0 0   0   0   ... C_m | | S_m |   | F_m |
   *
   * We solve for blocks individually (U, P, T, S_*).
   * First, we solve for U and P (the Stokes part of the problem) by using a Schur Complement approach
   *    B A^-1 B^T P = B A^-1 F - G --> Solve for P
   *    A U = F - B^T P ---> Solve for U
   */
  template<int dim>
  void CRFProblem<dim>::solve() {
    Timer timer;

    // First solve for velocities (U) and pressures (P)
    const auto &A = system_matrix.block(0, 0);
    const auto &B = system_matrix.block(1, 0);

    const auto &F = system_rhs.block(0);
    const auto &G = system_rhs.block(1);

    auto &U = solution.block(0);
    auto &P = solution.block(1);

    // Representations of our linear operators
    const auto op_A = linear_operator(A);
    const auto op_B = linear_operator(B);

    // Form the preconditioner for A
    SparseILU<double> preconditioner_A;
    preconditioner_A.initialize(system_matrix.block(0, 0), SparseILU<double>::AdditionalData());

    SolverControl solver_control_A(solution.block(0).size() * 10, 1.e-6 * system_rhs.block(0).l2_norm());
    SolverCG<Vector<double>> solver_A(solver_control_A);

    const auto op_A_inv = inverse_operator(op_A, solver_A, preconditioner_A);

    // Schur Complement
    const auto op_schur = op_B * op_A_inv * transpose_operator(op_B);

    const auto schur_rhs = op_B * op_A_inv * F - G;

    SolverControl solver_control_schur(P.size() * 10, 1.e-12 * system_rhs.block(1).l2_norm());
    SolverGMRES<Vector<double>> solver_schur(solver_control_schur);

    // Form the preconditioner for the Schur complement. This will be the inverse of the pressure mass matrix.
    auto &M = preconditioner_matrix.block(1,1);
    const auto op_M = linear_operator(M);

    SparseILU<double> M_preconditioner;
    M_preconditioner.initialize(M, SparseILU<double>::AdditionalData());

    SolverControl solver_control_M(solution.block(1).size(), 1.e-6 * system_rhs.block(1).l2_norm());
    SolverCG<Vector<double>> solver_M(solver_control_M);

    const auto schur_preconditioner = inverse_operator(op_M, solver_M, M_preconditioner);

    const auto op_schur_inv = inverse_operator(op_schur, solver_schur, schur_preconditioner);

    std::cout << "Solving for pressure..." << std::flush;
    P = op_schur_inv * schur_rhs;
    std::cout << " done... ";

    std::cout << solver_control_schur.last_step()
              << " iterations to obtain convergence..."
              << std::endl << std::flush;

    // Solve for U
    std::cout << "Solving for velocity..." << std::flush;
    U = op_A_inv * (F - transpose_operator(op_B) * P);
    std::cout << " done... ";

    std::cout << solver_control_A.last_step()
              << " iterations to obtain convergence."
              << std::endl << std::flush;


    // Solver for advection equations
    {
      SolverControl solver_control_temp(solution.block(2).size() * 10, 1.e-6 * system_rhs.block(2).l2_norm());
      SolverGMRES<Vector<double>> solver_temp(solver_control_temp);

      // Solve for temperature
      std::cout << "Solving for temperature..." << std::flush;

      solver_temp.solve(system_matrix.block(2, 2),
                             solution.block(2),
                             system_rhs.block(2),
                             PreconditionIdentity());

      std::cout << " done... ";

      std::cout << solver_control_temp.last_step()
                << " iterations to obtain convergence."
                << std::endl << std::flush;
    }

    // Solve for chemical species
    for (unsigned int chem = 0; chem < n_rxns; ++chem)
    {
      std::cout << "Chemcial " << chem << " rhs l2 norm = " << system_rhs.block(3+chem).l2_norm() << std::endl;
      SolverControl solver_control_chem(solution.block(3+chem).size() * 10,
                                        1.e-6 * system_rhs.block(3+chem).l2_norm());
      SolverGMRES<Vector<double>> solver_chem(solver_control_chem);

      std::cout << "Solving for chemical species " << chem << "..." << std::flush;

      solver_chem.solve(system_matrix.block(3+chem, 3+chem),
                        solution.block(3+chem),
                        system_rhs.block(3+chem),
                        PreconditionIdentity());

      std::cout << " done... ";

      std::cout << solver_control_chem.last_step()
                << " iterations to obtain convergence."
                << std::endl << std::flush;
      }

    constraints.distribute(solution);

    timer.stop();
    std::cout << "done (" << timer.cpu_time() << "s)" << std::endl;
  }


  // FIXME: make chemical names an input for a nicer output
  template<int dim>
  void CRFProblem<dim>::output_results(const unsigned int cycle) const {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    solution_names.emplace_back("temperature");
    for (unsigned int chem = 0; chem < n_rxns; ++chem) {
      std::string label = "Chemical_" + std::to_string(chem + 1);
      solution_names.emplace_back(label);
    }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(dim,
                                      DataComponentInterpretation::component_is_part_of_vector);
    // pressure
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    // temperature
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    // chemical species
    for (unsigned int chem = 0; chem < n_rxns; ++chem) {
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    }

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::ofstream output("solution-" + Utilities::int_to_string(cycle, 2) + ".vtk");
    data_out.write_vtk(output);
  }


  template<int dim>
  void CRFProblem<dim>::refine_mesh() {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    // The Kelly error estimator tells us where interpolation error is high.
    // This does not give an appropriate estimate of the actual error, but it gives a good indication
    // of where the mesh should be refined. In order to take all equations into consideration, we need
    // to normalize the Kelly error from each equation and add them together.
    // Pressure error gives a good indication of velocity error in the Stokes equation, so we ignore velocity.
    // All advection terms are calculated.
    // The error from each equation is normalized by the L infinity norm of the error on each cell.
    {
      Vector<float> pressure_error_per_cell(triangulation.n_active_cells());
      FEValuesExtractors::Scalar pressure(dim);
      KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(stokes_degree + 1),
                                         std::map<types::boundary_id, const Function<dim> *>(), solution,
                                         pressure_error_per_cell, fe.component_mask(pressure));

      double max_error = pressure_error_per_cell.linfty_norm();
      if (max_error == 0.)
        max_error = 1.;

      for (unsigned int cell = 0; cell < triangulation.n_active_cells(); ++cell) {
        estimated_error_per_cell[cell] += pressure_error_per_cell[cell] / max_error;
      }
    }

    {
      Vector<float> temperature_error_per_cell(triangulation.n_active_cells());
      FEValuesExtractors::Scalar temperature(dim + 1);
      KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(stokes_degree + 1),
                                         std::map<types::boundary_id, const Function<dim> *>(), solution,
                                         temperature_error_per_cell, fe.component_mask(temperature));

      double max_error = temperature_error_per_cell.linfty_norm();
      if (max_error == 0.)
        max_error = 1.;

      for (unsigned int cell = 0; cell < triangulation.n_active_cells(); ++cell) {
        estimated_error_per_cell[cell] += temperature_error_per_cell[cell] / max_error;
      }
    }

    for (unsigned int chem = 0; chem < n_rxns; ++chem) {
      Vector<float> chemical_error_per_cell(triangulation.n_active_cells());
      FEValuesExtractors::Scalar chemical(dim + 2 + chem);
      KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(stokes_degree + 1),
                                         std::map<types::boundary_id, const Function<dim> *>(), solution,
                                         chemical_error_per_cell, fe.component_mask(chemical));

      double max_error = chemical_error_per_cell.linfty_norm();
      if (max_error == 0.)
        max_error = 1.;

      for (unsigned int cell = 0; cell < triangulation.n_active_cells(); ++cell) {
        estimated_error_per_cell[cell] += chemical_error_per_cell[cell] / max_error;
      }
    }

    GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.0);

    triangulation.execute_coarsening_and_refinement();
  }
}



/*
 * MAIN FUNCTION
 */

int main()
{
    try
    {
        using namespace CRF;

        // 2D problem
        // Stokes: Q2-Q1
        // Advection: Q1
        // Chemical reaction: 2H_2 + O_2 -> 2H_2O
        // Viscosity of N2 = 1.66 according to Google (at 1.76 degree C, but ignore that for now)
        CRFProblem<2> crf_problem(1, 1, 4, 1.66);
        crf_problem.run(3);
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << "Aborting!" << std::endl
                  << "------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}