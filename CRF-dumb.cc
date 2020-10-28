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
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
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
#include <memory>



namespace CRF
{
    using namespace dealii;


    /*
     * CRF CLASS -- Declaration
     */

    // FIXME: Do I need to add rxns as a template parameter?
    template<int dim>
    class CRFProblem
    {
    public:
        CRFProblem(const unsigned int stokes_degree,
            const unsigned int advect_degree);

        void run();

    private:
        void setup_dofs();

        void assemble_system();

        void solve();

        void output_results() const;

        const unsigned int stokes_degree;
        const unsigned int advect_degree;

        Triangulation <dim> triangulation;
        FESystem<dim>       fe;
        DoFHandler<dim>     dof_handler;

        AffineConstraints<double> constraints;

        BlockSparsityPattern        sparsity_pattern;
        BlockSparseMatrix<double>   system_matrix;

        BlockVector<double> solution;
        BlockVector<double> system_rhs;
    };


    /*
     * RIGHT HAND SIDE FUNCTION -- Declaration
     */

    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
        // input = dim
        // output = dim + 1 + 1 + rxns --> velocity + pressure + temperature + chemical species
        // FIXME: rxns hardcoded at the moment!
        RightHandSide()
            : Function<dim>(dim + 1 + 1 + 2)
        {}

        // we need to write our own function for each component of the right hand side
        // Inputs:
        // The spatial coordinates we're evaluating at
        // The component we want to evaluate
        virtual double value(const Point<dim> & p,
                             const unsigned int component = 0) const override;

        // we need to write a custom function to output the entire vector
        // Inputs:
        // The spatial coordinates we're evaluating at
        // An empty vector to store the result
        virtual void vector_value(const Point<dim> &p,
                                  Vector<double> &values) const override;
    };



    /*
     * ADVECTION FIELD -- Declaration
     */

    template <int dim>
    class AdvectionField : public TensorFunction<1, dim>
    {
    public:
        virtual Tensor<1, dim> value(const Point<dim> &p) const override;

        DeclException2(ExcDimensionMismatch,
                       unsigned int,
                       unsigned int,
                       << "The vector has size " << arg1 << " but should have "
                       << arg2 << " elements.");
    };



    /*
     * BOUNDARY VALUES -- Declaration
     */

    // FIXME: Make n_rxns template parameter?
    template <int dim>
    class BoundaryValues : public Function<dim>
    {
    public:
        BoundaryValues()
            : Function<dim>(dim + 1 + 1 + 2)
        {}

        virtual double value(const Point<dim> & p,
                             const unsigned int component = 0) const override;

        virtual void vector_value(const Point<dim> &p,
                                  Vector<double> & values) const override;
    };


    /*
     * CRF CLASS -- Implementation
     */


    // Constructor
    // FIXME: Hardcoded number of chemical species. Can I do: FE_Q<dim>(advect_degree), n_rxns?
    template<int dim>
    CRFProblem<dim>::CRFProblem(const unsigned int stokes_degree,
                  const unsigned int advect_degree)
        : stokes_degree(stokes_degree)
        , advect_degree(advect_degree)
        , triangulation(Triangulation<dim>::maximum_smoothing)
        , fe(FE_Q<dim>(stokes_degree+1), dim, // velocity
             FE_Q<dim>(stokes_degree), 1,     // pressure
             FE_Q<dim>(advect_degree), 1,     // temperature
             FE_Q<dim>(advect_degree), 1,     // chemical species 1
             FE_Q<dim>(advect_degree), 1)     // chemical species 2
        , dof_handler(triangulation)
    {}



    template<int dim>
    void CRFProblem<dim>::run()
    {
        // Attach a grid to the triangulation
        {
            std::vector<unsigned int> subdivisions(dim, 1);
            subdivisions[0] = 4;

            const Point<dim> bottom_left = (dim == 2 ?
                                                Point<dim>(-2, -1) :        // 2d case
                                                Point<dim>(-2, 0, -1));     // 3d case

            const Point<dim> top_right   = (dim == 2 ?
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
        triangulation.refine_global(4 - dim);

        // Solve the problem
        // FIXME: Once mesh refinement is added, wrap in a for loop over desired number of cycles
        setup_dofs();

        std::cout << "Assembling..." << std::endl << std::flush;
        assemble_system();

        std::cout << "Solving..." << std::flush;
        solve();

        output_results();

        std::cout << std::endl;
    }



    template<int dim>
    void CRFProblem<dim>::setup_dofs()
    {
        dof_handler.distribute_dofs(fe);
        DoFRenumbering::Cuthill_McKee(dof_handler);

        // Renumber the DoF's so that solution variables are ordered in blocks
        // FIXME: hardcoding of number of reactions
        // Velocities = block 0, pressure = block 1, Temperature = block 2, Chemical 1 = block 3, Chemical 2 = block 4
        std::vector<unsigned int> block_component(dim + 1 + 1 + 2, 0);
        block_component[dim] = 1;
        block_component[dim + 1] = 2;
        block_component[dim + 2] = 3;
        block_component[dim + 3] = 4;
        DoFRenumbering::component_wise(dof_handler, block_component);

        // Set up hanging node and Dirichlet BC constraints
        {
            constraints.clear();

            FEValuesExtractors::Vector velocities(0);
            DoFTools::make_hanging_node_constraints(dof_handler, constraints);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     1,
                                                     BoundaryValues<dim>(),
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
                  for (unsigned int block = 1; block<dofs_per_block.size(); ++block)
                  {
                      std::cout << '+' << dofs_per_block[block];
                  }
                  std::cout << ')' << std::endl;

        {
            // FIXME: Hardcoded number of blocks
            BlockDynamicSparsityPattern dsp(5, 5);

            for (unsigned int row = 0; row < 5; ++row)
                for (unsigned int col = 0; col < 5; ++col)
                {
                    dsp.block(row, col).reinit(dofs_per_block[row], dofs_per_block[col]);
                }

            dsp.collect_sizes();

            DoFTools::make_sparsity_pattern(dof_handler,
                                            dsp,
                                            constraints,
                                            false);

            sparsity_pattern.copy_from(dsp);
        }

        system_matrix.reinit(sparsity_pattern);

        solution.reinit(dofs_per_block.size());
        system_rhs.reinit(dofs_per_block.size());
        for (unsigned int block = 0; block < dofs_per_block.size(); ++block)
        {
            solution.block(block).reinit(dofs_per_block[block]);
            system_rhs.block(block).reinit(dofs_per_block[block]);
        }
        solution.collect_sizes();
        system_rhs.collect_sizes();
    }



    template<int dim>
    void CRFProblem<dim>::assemble_system()
    {
        system_matrix   = 0;
        system_rhs      = 0;

        // FIXME: why degree+2?
        QGauss<dim> quadrature_formula(stokes_degree+2);

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
                                         QGauss<dim - 1>(advect_degree+1),
                                         update_values | update_quadrature_points |
                                         update_JxW_values | update_normal_vectors);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;

        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points  = fe_face_values.get_quadrature().size();

        FullMatrix<double>  local_matrix(dofs_per_cell, dofs_per_cell);

        Vector<double>      local_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        const RightHandSide<dim>    right_hand_side;
        std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim+4));

        const BoundaryValues<dim>   boundary_value;
        // FIXME: create vector to hold boundary values at each quadrature point??

        AdvectionField<dim> advection_field;

        // FIXME: Hardcoded number of chemicals
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);
        const FEValuesExtractors::Scalar temperature(dim+1);
        const FEValuesExtractors::Scalar chemical1(dim+2);
        const FEValuesExtractors::Scalar chemical2(dim+3);

        std::vector<SymmetricTensor<2, dim>>    symgrad_phi_u(dofs_per_cell);
        std::vector<double>                     div_phi_u(dofs_per_cell);
        std::vector<double>                     phi_p(dofs_per_cell);
        std::vector<Tensor<1, dim>>             grad_phi_T(dofs_per_cell);
        std::vector<double>                     phi_T(dofs_per_cell);
        // FIXME: Hardcoded number of chemicals
        std::vector<Tensor<1, dim>>             grad_phi_s1(dofs_per_cell);
        std::vector<double>                     phi_s1(dofs_per_cell);
        std::vector<Tensor<1, dim>>             grad_phi_s2(dofs_per_cell);
        std::vector<double>                     phi_s2(dofs_per_cell);

        // Loop over all cells
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
            local_matrix    = 0;
            local_rhs       = 0;

            right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                              rhs_values);

            // Calculate delta for streamline diffusion based off cell diameter
            const double delta = 0.1 * cell->diameter();

            // loop over all quadrature points
            for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                    // We are populating a matrix with (phi_i, phi_j) where ( , ) describes
                    // the complete bilinear form of our weak form.
                    // Since we loop over i and then j, we can save computational energy by
                    // precalculating all phi terms and then just call them within the loop.
                    symgrad_phi_u[k]    = fe_values[velocities].symmetric_gradient(k, q);
                    div_phi_u[k]        = fe_values[velocities].divergence(k, q);
                    phi_p[k]            = fe_values[pressure].value(k, q);
                    grad_phi_T[k]       = fe_values[temperature].gradient(k, q);
                    phi_T[k]            = fe_values[temperature].value(k, q);
                    // FIXME: Hardcoded number of chemicals
                    grad_phi_s1[k]      = fe_values[chemical1].gradient(k, q);
                    phi_s1[k]           = fe_values[chemical1].value(k, q);
                    grad_phi_s2[k]      = fe_values[chemical2].gradient(k, q);
                    phi_s2[k]           = fe_values[chemical2].value(k, q);
                }

                // Value of JxW for quadrature point
                const double dx = fe_values.JxW(q);

                // Evaluate advection field at quadrature point
                const Tensor<1, dim> advection_direction = advection_field.value(fe_values.quadrature_point(q));

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        local_matrix(i, j) +=
                                (
                                        2 * (symgrad_phi_u[i] * symgrad_phi_u[j])
                                        - div_phi_u[i] * phi_p[j]
                                        - phi_p[i] * div_phi_u[j]
                                        + (phi_T[i] + delta * (advection_direction * grad_phi_T[i])) *
                                          (advection_direction * grad_phi_T[j])
                                        + (phi_s1[i] + delta * (advection_direction * grad_phi_s1[i])) *
                                          (advection_direction * grad_phi_s1[j])
                                        + (phi_s2[i] + delta * (advection_direction * grad_phi_s2[i])) *
                                          (advection_direction * grad_phi_s2[j])
                                ) * dx;
                    }

                    // Shape functions are only non-zero in one component
                    const unsigned int component_i = fe.system_to_component_index(i).first;
                    local_rhs(i) += (fe_values.shape_value(i, q) * rhs_values[q](component_i)) * dx;
                    //FIXME: Letting Neumann BC = 0 for now
                }
            }
                // We now check to see if any face on the cell is on an inflow boundary.
                // If so, the local matrix and rhs get contributions from the advection equations.
                for (const auto &face : cell->face_iterators())
                {
                    if (face->at_boundary())
                    {
                        fe_face_values.reinit(cell, face);

                        // Loop over quadrature points and see if each point is on inflow or outflow direction.
                        // The scalar product between the advection direction and the normal direction should
                        // be negative on the inflow boundary. This is because the normal direction points outside
                        // the domain while the advection direction points inside.
                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                        {
                            // define variables to make things easier to read
                            const Point<dim> q_point = fe_face_values.quadrature_point(q);
                            const Tensor<1, dim> advection_direction = advection_field.value(q_point);
                            const Tensor<1, dim> normal_vector       = fe_face_values.normal_vector(q);
                            if (advection_direction * normal_vector < 0.)
                            {
                                // If the face is part of the inflow boundary, compute the contributions to the
                                // local matrix and right hand side.
                                const double dx = fe_face_values.JxW(q);

                                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                {
                                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                                    {
                                        local_matrix(i, j) -=
                                                (
                                                    (advection_direction * normal_vector)
                                                    * (
                                                        phi_T[i] * phi_T[j]
                                                        + phi_s1[i] * phi_s1[j]
                                                        + phi_s2[i] * phi_s2[j]
                                                    )
                                                ) * dx;
                                    }

                                    // FIXME: Hardcoded boundary value component
                                    // FIXME: Precalculate boundary values??
                                    double test1 = advection_direction * normal_vector;
                                    double test2 = boundary_value.value(q_point, dim + 1);
                                    local_rhs(i) -=
                                            (
                                                    (advection_direction * normal_vector)
                                                    * (
                                                            phi_T[i] * boundary_value.value(q_point, dim + 1)
                                                            + phi_s1[i] * boundary_value.value(q_point, dim + 2)
                                                            + phi_s2[i] * boundary_value.value(q_point, dim + 3)
                                                            )
                                                    ) * dx;
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
        }
    }



    template <int dim>
    void CRFProblem<dim>::solve()
    {
        // To keep things simple for the moment, we simply use a direct solver.
        std::cout << "Solving linear system... ";
        Timer timer;

        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(solution, system_rhs);

        timer.stop();
        std::cout << "done (" << timer.cpu_time() << "s)" << std::endl;
    }



    template <int dim>
    void CRFProblem<dim>::output_results() const
    {
        // FIXME: hardcoded number of chemicals
        std::vector<std::string> solution_names(dim, "velocity");
        solution_names.emplace_back("pressure");
        solution_names.emplace_back("temperature");
        solution_names.emplace_back("chemical_1");
        solution_names.emplace_back("chemical_2");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
                data_component_interpretation(dim,
                                              DataComponentInterpretation::component_is_part_of_vector);
        // pressure
        data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        // temperature
        data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        // chemical species 1
        data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        // chemical species 2
        data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution,
                                 solution_names,
                                 DataOut<dim>::type_dof_data,
                                 data_component_interpretation);
        data_out.build_patches();

        std::ofstream output("solution.vtk");
        data_out.write_vtk(output);
    }



    /*
     * RIGHT HAND SIDE FUNCTION -- Implementation
     */

    // FIXME: Some hardcoded stuff
    template<int dim>
    double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int component) const
    {
        Assert((component >= 0) && (component <= dim+1+2), ExcIndexRange(component, 0, dim+1+2+1));
        // component = 0-(dim-1)    --> f_a
        // component = dim          --> f_b
        // component = dim + 1      --> f_T
        // component = dim + 2      --> f_1
        // ...
        // component = dim + m + 1  --> f_m

        // FIXME: For now, we just use zero rhs for simplicity
        if (component < dim)
            return 0.;
        else if (component == dim)
            return 0.;
        else if (component == dim+1)
            return 0.;
        else if (component == dim+2)
        {
            // FIXME: hardcoded for testing
            const Point<2> center_point(0.0, -0.5);
            if ((p - center_point).norm_square() < 0.25)
                return 1.;
            else
                return 0.;
        }
        else
            return 0.;
    }

    // FIXME: Do I even need this?
    template <int dim>
    void RightHandSide<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
    {
        for (unsigned int c = 0; c < this->n_components; ++c)
            values[c] = RightHandSide<dim>::value(p, c);
    }


    /*
     * ADVECTION FIELD -- Implementation
     */

    template <int dim>
    Tensor<1, dim> AdvectionField<dim>::value(const Point<dim> &p) const
    {
        Point<dim> value;
        // FIXME: For now we just let the advection field be 1 in the y-direction
        if (dim == 1)
            value[0] = 1.;
        else
        {
            for (unsigned int i = 0; i < dim; ++i)
            {
                if (i == 1)
                    value[i] = 1.;
                else
                    value[i] = 0.;
            }
        }

        return value;
    }



    /*
     * BOUNDARY VALUES -- Implementation
     */

    template <int dim>
    double BoundaryValues<dim>::value(const Point<dim> & p,
                                      const unsigned int component) const
    {
        Assert(component < this->n_components,
               ExcIndexRange(component, 0, this->n_components));

        // velocities away from origin
        if (component < dim)
            return (p[component] < 0 ? -1 : (p[component] > 0 ? 1 : 0));
        else if (component == dim)
            return 0.; // zero BC for pressure
        else
        {
            // Option 1 -- non-zero only on small part of boundary.
            // Use this alongside no right hand side to check if value
            // is advected along streamline only.
            // Our domain is
            //  ________________
            // |                |
            // |                |
            // |                |
            // |________________|
            //
            if ( p[0] < .1 && p[0] > -0.1)
                return 0.;
            else
                return 0.;

            // Option 2 -- zero along entire boundary.
            // Use this alongside a right hand side to check if the source/sink
            // within the domain interacts with streamline correctly.
            // return 0.;
        }
    }

    // FIXME: Do I even need this??
    template <int dim>
    void BoundaryValues<dim>::vector_value(const Point<dim> &p, Vector<double> & values) const
    {
        for (unsigned int c = 0; c < this->n_components; ++c)
            values[c] = BoundaryValues<dim>::value(p, c);
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
        CRFProblem<2> crf_problem(1, 1);
        crf_problem.run();
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