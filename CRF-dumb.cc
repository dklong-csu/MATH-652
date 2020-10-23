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

/*  TABLE OF CONTENTS
 *
 * (1) HEADER FILES
 * (2) CRF CLASS
 * (3) RIGHT HAND SIDE FUNCTION
 *
 *
 *
 *
 */



/*
 * HEADER FILES
 */

// include necessary deal.II header files
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
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
    class CRF
    {
    public:
        CRF(const unsigned int stokes_degree,
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
                                  Vector<double> &value) const override;
    };



    /*
     * CRF CLASS -- Implementation
     */


    // Constructor
    // FIXME: Hardcoded number of chemical species. Can I do: FE_Q<dim>(advect_degree), n_rxns?
    template<int dim>
    CRF<dim>::CRF(const unsigned int stokes_degree,
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
    void CRF<dim>::run()
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
    void CRF<dim>::setup_dofs()
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

            // FIXME: Hardcoded number of equations
            Table<2, DoFTools::Coupling> coupling(dim+1+1+2, dim+1+1+2);
            // Eq 0-(dim-1) -- u_{x_i}, p
            // Eq dim -- u_{x_1},...,u_{x_dim}
            // Eq dim+1 -- T
            // Eq dim+2 -- S_1
            // Eq dim+3 -- S_2
            // ...
            for (unsigned int row = 0; row < dim+1+1+2; ++row)
                for(unsigned int col = 0; col < dim+1+1+2; ++col)
                {
                    if (row < dim)
                    {
                        if ((row == col)||(col==dim))
                            coupling[row][col] = DoFTools::always;
                        else
                            coupling[row][col] = DoFTools::none;
                    }
                    else if (row == dim)
                    {
                        if (col < dim)
                            coupling[row][col] = DoFTools::always;
                        else
                            coupling[row][col] = DoFTools::none;
                    }
                    else
                    {
                        if (col == row)
                            coupling[row][col] = DoFTools::always;
                        else
                            coupling[row][col] = DoFTools::none;
                    }
                }

            DoFTools::make_sparsity_pattern(dof_handler,
                                            coupling,
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
    void CRF<dim>::assemble_system()
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

    }







    void solve();

    void output_results() const;

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
            return 0;
        else if (component == dim)
            return 0;
        else if (component == dim+1)
            return 0;
        else if (component == dim+2)
            return 0;
        else
            return 0;
    }
}


























