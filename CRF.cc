// This is a modification/combination of step-22 and step-9 of deal.II
/* This program works on the simplest case
 * mu div( e(u) ) + grad( p ) = 0 --> stokes
 * - div( u ) = 0 --> stokes
 * B.grad( T ) = 0 --> advection
 *
 * Future work will address coupling the wind speed with the stokes equation (i.e. B = u ),
 * adding additional advection equations to correspond to chemical reactions,
 * and adding non-homogeneity which corresponds to the chemical reaction.
 *
 * For the chemical reactions, we will consider a combustion reaction
 * The general form of a combustion reaction is:
 * hydrocarbon + oxygen -> carbon dioxide + water
 * source: https://www.thoughtco.com/combustion-reactions-604030
*/

/* TABLE OF CONTENTS
 *
 * 1. HEADER FILES
 * 2. PRECONDITIONER
 * 3. BOUNDARY CONDITIONS
 * 4. RIGHT HAND SIDE
 * 5. ADVECTION FIELD
 * 6. INVERSE MATRICES
 * 7. SCHUR COMPLEMENT
 * 8. CHEMICALLY REACTING FLOWS PROBLEM
 */

/*
 * HEADER FILES
 */

// As usual, we start by including some well-known files:
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

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>

// Then we need to include the header file for the sparse direct solver
// UMFPACK:
#include <deal.II/lac/sparse_direct.h>

// This includes the library for the incomplete LU factorization that will be
// used as a preconditioner in 3D:
#include <deal.II/lac/sparse_ilu.h>

// This is C++:
#include <iostream>
#include <fstream>
#include <memory>

// Run in parallel
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>

// For the advection field
#include <deal.II/base/tensor_function.h>



namespace CRF
{
    using namespace dealii;

    /* PRECONDITIONER
     *
     * The Stokes problem requires a preconditioner, but we want to use different
     * preconditioners based on the problem dimension. We create a template which
     * encodes this choice so that our future code is dimension independent.
     *
     */
    template <int dim>
    struct InnerPreconditioner;

    // 2D -> sparse direct solver
    template <>
    struct InnerPreconditioner<2>
    {
        using type = SparseDirectUMFPACK;
    };

    // 3D -> use incomplete LU factorization (ILU)
    template <>
    struct InnerPreconditioner<3>
    {
        using type = SparseILU<double>;
    };



    /* BOUNDARY CONDITIONS
     *
     * Here we declare and implement boundary conditions for each state variable
     * We have:
     * velocity: u ->           dim-dimensional
     * pressure: p ->           1-dimensional
     * temperature: T ->        1-dimensional
     * chemical species: S ->   rxns-dimensional (each chemical is 1-dimensional)
     *
     */

    /*
     * DECLARATIONS
     */

    // Velocity BC declaration
    template <int dim>
    class BoundaryValuesVelocity : public Function<dim>
    {
    public:
        // FIXME: the Stokes problem is <dim>(dim + 1) -- why?
        // FIXME: is it: <dim>=input dimension; (dim + 1 ) = output dimension?
        BoundaryValuesVelocity()
            : Function<dim>(dim)
        {}

        virtual double value(const Point<dim> & p,
                             const unsigned int component = 0) const override;

        virtual void vector_value(const Point<dim> & p,
                                  Vector<double> &  values) const override;
    };


    // Pressure BC declaration
    template <int dim>
    class BoundaryValuesPressure : public Function<dim>
    {
    public:
        // the output is always a scalar
        BoundaryValuesPressure()
            : Function<dim>(1)
        {}

        virtual void value(const Point<dim> & p,
                           Vector<double> &  values) const override;
    };

    // Temperature BC declaration
    template <int dim>
    class BoundaryValuesTemperature : public Function<dim>
    {
    public:
        // the output is always a scalar
        BoundaryValuesTemperature()
                : Function<dim>(1)
        {}

        virtual void value(const Point<dim> & p,
                           Vector<double> &  values) const override;
    };

    // Chemical species BC declaration
    template <int dim, int rxns>
    class BoundaryValuesChemicals : public Function<dim>
    {
    public:
        BoundaryValuesChemicals()
                : Function<dim>(rxns)
        {}

        virtual double value(const Point<dim> & p,
                             const unsigned int component = 0) const override;

        virtual void vector_value(const Point<dim> & p,
                                  Vector<double> &  values) const override;
    };

    /*
     * IMPLEMENTATIONS
     */

    // Velocity BC implementation
    // Evaluate a specified component
    template<int dim>
    double BoundaryValuesVelocity<dim>::value(const Point<dim> & p,
                                              const unsigned int component) const
    {
        // Check to make sure the requested component is within our spatial dimension
        Assert(component < this->n_components,
               ExcIndexRange(component, 0, this->n_components));

        // Dirichlet boundary condtions for velocity:
        // u_x(boundary) = step function -> i.e. u_x(<0)= -1, u_x(0) = 0, u_x(>0) = 1
        if (component == 0)
            return (p[0] < 0 ? -1 : (p[0] > 0 ? 1 : 0));
        // all other velocity components are zero on the boundary
        return 0;
    }

    // Return all components of the velocity BC
    template <int dim>
    void BoundaryValuesVelocity<dim>::vector_value(const Point<dim> &p,
                                                   Vector<double> &  values) const
    {
        // loop over all components of the velocity output
        for (unsigned int c = 0; c < this->n_components; ++c)
            values(c) = BoundaryValuesVelocity<dim>::value(p, c);
    }


    // Pressure BC implementation
    template <int dim>
    void BoundaryValuesPressure<dim>::value(const Point<dim> &p,
                                            Vector<double> &  values) const
    {
        // Homogenous BC
        values(0) = 0;
    }

    // Temperature BC implementation
    template <int dim>
    void BoundaryValuesPressure<dim>::value(const Point<dim> &p,
                                            Vector<double> &  values) const
    {
        // Flux on inflow boundary
        values(0) = 1;
    }

    // Chemical species BC implementation
    // Evaluate a specified component
    template<int dim, int rxns>
    double BoundaryValuesChemicals<dim, rxns>::value(const Point<dim> & p,
                                                     const unsigned int component) const
    {
        // Check to make sure the requested component is within our spatial dimension
        Assert(component < this->n_components,
               ExcIndexRange(component, 0, this->n_components));

        // Neumann boundary condtions for chemical species
        // 1 for all chemicals
        return 1;
    }

    // Return all components of the chemical species BC
    template <int dim, int rxns>
    void BoundaryValuesChemicals<dim, rxns>::vector_value(const Point<dim> &p,
                                                          Vector<double> &  values) const
    {
        // loop over all chemical species
        for (unsigned int chem = 0; chem < this->n_components; ++chem)
            values(chem) = BoundaryValuesChemicals<dim, rxns>::value(p,chem);
    }



    /* RIGHT HAND SIDE
     *
     * Here we implement the right hand side for our equations.
     * Stokes equation:
     *      -mu div( e(u) ) + grad( p ) = 0
     *      !! we will continue to use 0 right hand side for this
     *      -div( u ) = 0
     *      FIXME: we eventually want the right hand side to encode the thermal expansion of the gas as reactions proceed
     *
     * Advection equation -- temperature
     *      beta . grad( T ) = 0
     *      FIXME: eventually we want the rhs to result from heat production due to chemical reactions
     *
     * Advection equations -- chemical species
     *      beta . grad( S_j ) = 0
     *      FIXME: eventually the rhs will be based on the creation or consumption of this species due to the reactions
     *
     */

    /*
     * DECLARATIONS
     */

    // Stokes right hand side declaration
    template <int dim>
    class RightHandSideStokes : public Function<dim>
    {
    public:
        // Stokes equation maps R^(dim)->R^(dim+1)
        RightHandSideStokes()
            : Function<dim>(dim+1)
        {}

        // we want to describe the right hand side for each component
        virtual double value(const Point<dim> & p,
                             const unsigned int component = 0) const override;

        // we want a way to access all rhs for a particular point in a single data structure
        virtual void vector_value(const Point<dim> & p,
                                  Vector<double> & value) const override;
    };


    // Temperature advection right hand side declaration
    template <int dim>
    class RightHandSideTemperature : public Function<dim>
    {
    public:
        virtual double value(const Point<dim> & p,
                             const unsigned int component = 0) const override;
    };

    // Chemical species advection right hand side declaration
    // FIXME: For now, we're just using one function since all have the same rhs.
    // FIXME: When rhs is added, a class for each species will be required.
    template <int dim>
    class RightHandSideChemical : public Function<dim>
    {
    public:
        virtual double value(const Point<dim> & p,
                             const unsigned int component = 0) const override;
    };

    /*
     * IMPLEMENTATIONS
     */

    // Stokes right hand side implementation
    template <int dim>
    double RightHandSideStokes<dim>::value(const Point<dim> & p,
                                           const unsigned int component) const
    {
        return 0;
    }

    template <int dim>
    void RightHandSideStokes<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> &  values) const
    {
        for (unsigned int c = 0; c < this->n_components; ++c)
            values(c) = RightHandSideStokes<dim>::value(p, c);
    }

    // Temperature advection implementation
    template <int dim>
    double RightHandSideTemperature<dim>::value(const Point<dim> & p,
                                                const unsigned int component) const
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));
        return 0;
    }

    // Chemical species advection implementation
    template <int dim>
    double RightHandSideChemical<dim>::value(const Point<dim> & p,
                                                const unsigned int component) const
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));
        return 0;
    }

    /*ADVECTION FIELD
     *
     * This is a class which describes the advection field.
     * This will be a vector with a component for each space dimension.
     * The temperature and chemical species exist in the same medium, so
     * they share an advection field.
     * FIXME: We eventually want to couple the advection equations to the Stokes problem
     * FIXME: in which case the velocity solution of Stokes will be the advection field.
     * FIXME: But, for now, we'll just have a simple field which is 1 in each direction.
     *
     */

    /*
     * DECLARATION
     */

    template<int dim>
    class AdvectionField : public TensorFuntion<1, dim>
    {
    public:
        virtual Tensor<1, dim> value(const Point<dim> &p) const override;

        // Declare an exception
        DeclException2(ExcDimensionMismatch,
                       unsigned int,
                       unsigned int,
                       << "The vector has size " << arg1 << " but should have "
                               << arg2 << " elements.");
    };

    /*
     * IMPLEMENTATION
     */

    template <int dim>
    Tensor<1, dim> AdvectionField<dim>::value(const Point<dim> &p) const
    {
        Point<dim> value;
        for (unsigned int i = 0; i < dim; ++i)
            value[i] = 0;
        return value;
    }


    /* INVERSE MATRICES
     *
     * This defines how we invert matrices.
     * Specifically, we need to use this for the Schur complement
     * which requires using A^-1 where A is an SPD matrix.
     * Hence vmult is defined using CG.
     * We use a small tolerance because we will have to perform
     * this operation often.
     *
     */
    // FIXME: I don't really understand what this is used for
    // Declaration of an InverseMatrix object
    template <class MatrixType, class PreconditionerType>
    class InverseMatrix : public Subscriptor
    {
    public:
        InverseMatrix(const MatrixType &        m,
                      const PreconditionerType &preconditioner);

        void vmult(Vector<double> &dst, const Vector<double> &src) const;

    private:
        const SmartPointer<const MatrixType>         matrix;
        const SmartPointer<const PreconditionerType> preconditioner;
    };


    template <class MatrixType, class PreconditionerType>
    InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
            const MatrixType &        m,
            const PreconditionerType &preconditioner)
            : matrix(&m)
            , preconditioner(&preconditioner)
    {}



    // Implementation of the vmult function
    template <class MatrixType, class PreconditionerType>
    void InverseMatrix<MatrixType, PreconditionerType>::vmult(
            Vector<double> &      dst,
            const Vector<double> &src) const
    {
        SolverControl            solver_control(src.size(), 1e-6 * src.l2_norm());
        SolverCG<Vector<double>> cg(solver_control);

        dst = 0;

        cg.solve(*matrix, dst, src, *preconditioner);
    }


    /* SCHUR COMPLEMENT
     *
     * The Stokes equation looks something like
     *
     * | A  B^T | U  = F
     * | B  0   | P  = G
     *
     * Expanded:
     * AU + B^TP = F (1)
     * BU = G        (2)
     *
     * We make the substitution U=B^-1G from (2) into (1) to get
     * AB^-1G + B^TP = F
     * and left multiply by BA^-1 to find
     * BA^-1B^TP = BA^-1F - G
     *
     * BA^-1B^T -- this is the Schur complement
     */

    // Class declaration
    template <class PreconditionerType>
    class SchurComplement : public Subscriptor
    {
    public:
        SchurComplement(
                const BlockSparseMatrix<double> &system_matrix,
                const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse);

        void vmult(Vector<double> &dst, const Vector<double> &src) const;

    private:
        const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
        const SmartPointer<
                const InverseMatrix<SparseMatrix<double>, PreconditionerType>>
                A_inverse;

        mutable Vector<double> tmp1, tmp2;
    };


    /* CONSTRUCTOR
     *
     * INPUTS:
     * system_matrix -- this is required to extract A and B
     * A_inverse -- this describes how we find A^-1 (i.e. CG)
     *
     */
    // FIXME: I'm confused how InverseMatrix knows to operate using system_matrix.block(0,0)
    template <class PreconditionerType>
    SchurComplement<PreconditionerType>::SchurComplement(
            const BlockSparseMatrix<double> &system_matrix,
            const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse)
            : system_matrix(&system_matrix)
            , A_inverse(&A_inverse)
            , tmp1(system_matrix.block(0, 0).m())
            , tmp2(system_matrix.block(0, 0).m())
    {}

    // This describes what happens when we multiply by the Schur complement
    // i.e. left multiply by B^T
    // then left multiply by A^-1
    template <class PreconditionerType>
    void
    SchurComplement<PreconditionerType>::vmult(Vector<double> &      dst,
                                               const Vector<double> &src) const
    {
        system_matrix->block(0, 1).vmult(tmp1, src);
        A_inverse->vmult(tmp2, tmp1);
        system_matrix->block(1, 0).vmult(dst, tmp2);
    }


    /* CHEMICALLY REACTING FLOWS PROBLEM
     *
     * Here we set up the main class of this program.
     * This will perform all of the setup steps to define the finite elements
     * as well as solving the problem and defining mesh refinement.
     *
     */

    // Class declaration
    template <int dim>
    class CRFProblem
    {
    public:
        CRFProblem(const unsigned int degree);
        void run();

    private:
        void setup_dofs();
        void assemble_system();
        void solve();
        void output_results(const unsigned int refinement_cycle) const;
        void refine_mesh();

        const unsigned int degree;

        Triangulation<dim> triangulation;
        FESystem<dim>      fe;
        DoFHandler<dim>    dof_handler;

        AffineConstraints<double> constraints;

        BlockSparsityPattern      sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;

        BlockSparsityPattern      preconditioner_sparsity_pattern;
        BlockSparseMatrix<double> preconditioner_matrix;

        BlockVector<double> solution;
        BlockVector<double> system_rhs;

        std::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;
    };

    // Constructor
    // Here we define the finite elements to be used.
    // For the Stokes equation, we need U to be Q^{degree + 1} and P to be Q^{degree}
    // For the rest of the advection equations, we use Q^{degree} elements
    template<int dim>
    CRFProblem<dim>::CRFProblem(const unsigned int degree)
        : degree(degree)
        , triangulation(Triangulation<dim>::maximum_smoothing)
        , fe(FE_Q<dim>(degree + 1), dim, /* U element */
             FE_Q<dim>(degree), 1,       /* P element */
             FE_Q<dim>(degree), 1,       /* T element */
             FE_Q<dim>(degree), 1,       /* S_1 element */
             FE_Q<dim>(degree), 1)       /* S_2 element */
    {}


    // setup_dofs
    // Given a mesh, this function associates degrees of freedom
    // and creates the associated matrices and vectors
    template <int dim>
    void CRFProblem<dim>::setup_dofs()
    {
        // FIXME: what does this do?
        A_preconditioner.reset();
        system_matrix.clear();
        preconditioner_matrix.clear();

        dof_handler.distribute_dofs(fe);
        DoFRenumbering::component_wise(dof_handler, block_component);

        // Dirichlet boundary conditions
        // FIXME: I don't really understand
        {
            constraints.clear();

            FEValuesExtractors::Vector velocities(0);
            DoFTools::make_hanging_node_constraints(dof_handler, constraints);
            // FIXME:
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     1,
                                                     BoundaryValues<dim>(),
                                                     constraints,
                                                     fe.component_mask(velocities));
        }

        constraints.close();

        const std::vector<types::global_dof_index> dofs_per_block =
                DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
        const unsigned int n_u = dofs_per_block[0];
        const unsigned int n_p = dofs_per_block[1];
        const unsigned int n_T = dofs_per_block[2];
        const unsigned int n_S1 = dofs_per_block[3];
        const unsigned int n_S2 = dofs_per_block[4];

        std::cout << "   Number of active cells: " << triangulation.n_active_cells()
                  << std::endl
                  << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                  << " (" << n_u << '+' << n_p << '+' << n_T << '+' << n_S1 << '+' << n_S2 << ')' << std::endl;

        // create the sparsity pattern object
        {
            BlockDynamicSparsityPattern dsp(5,5);

            std::vector<unsigned int> block_sizes = {n_u, n_p, n_T, n_S1, n_S2};

            for (unsigned int row = 0; row < block_sizes.size(); ++row)
            {
                for (unsigned int col = 0; col < block_sizes.size(); ++col)
                {
                    dsp.block(row, col).reinit(block_sizes[row], block_sizes[col]);
                }
            }

            dsp.collect_sizes();

            // FIXME: I do not understand what the 2 is used for
            Table<2, DoFTools::Coupling> coupling(dim+4, dim+4);

            for (unsigned int c = 0; c < dim + 4; ++c)
            {
                for (unsigned int d = 0; d < dim + 4; ++d)
                    if ((c < dim) && (d < dim))
                        coupling[c][d] = DoFTools::always;
                    else
                        coupling[c][d] = DoFTools::none;
            }

            DoFTools::make_sparsity_pattern(dof_handler,
                                            coupling,
                                            dsp,
                                            constraints,
                                            false);

            sparsity_pattern.copy_from(dsp);
        }


        // FIXME: Is this basically just saying which block needs a preconditioner?
        // FIXME: In our case we use a preconditioner on the Schur complement to solve for P?
        {
            BlockDynamicSparsityPattern preconditioner_dsp(5, 5);

            std::vector<unsigned int> block_sizes = {n_u, n_p, n_T, n_S1, n_S2};

            for (unsigned int row = 0; row < block_sizes.size(); ++row)
            {
                for (unsigned int col = 0; col < block_sizes.size(); ++col)
                {
                    preconditioner_dsp.block(row, col).reinit(block_sizes[row], block_sizes[col]);
                }
            }

            preconditioner_dsp.collect_sizes();

            Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 4, dim + 4);

            for (unsigned int c = 0; c < dim + 4; ++c)
                for (unsigned int d = 0; d < dim + 4; ++d)
                    if (((c == dim) && (d == dim)))
                        preconditioner_coupling[c][d] = DoFTools::always;
                    else
                        preconditioner_coupling[c][d] = DoFTools::none;

            DoFTools::make_sparsity_pattern(dof_handler,
                                            preconditioner_coupling,
                                            preconditioner_dsp,
                                            constraints,
                                            false);

            preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
        }

        // Create the system matrix, preconditioner matrix, solution vector, and rhs vector.
        system_matrix.reinit(sparsity_pattern);
        preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

        // FIXME: make the 5 and n_u, n_p, etc not as hard-coded and distributed throughout entire code
        solution.reinit(5);
        solution.block(0).reinit(n_u);
        solution.block(1).reinit(n_p);
        solution.block(2).reinit(n_T);
        solution.block(3).reinit(n_S1);
        solution.block(4).reinit(n_S2);
        solution.collect_sizes();

        system_rhs.reinit(5);
        system_rhs.block(0).reinit(n_u);
        system_rhs.block(1).reinit(n_p);
        system_rhs.block(2).reinit(n_T);
        system_rhs.block(3).reinit(n_S1);
        system_rhs.block(4).reinit(n_S2);
        system_rhs.collect_sizes();
    }


    // assemble_system
    template <int dim>
    void CRFProblem<dim>::assemble_system()
    {
        system_matrix           = 0;
        system_rhs              = 0;
        preconditioner_matrix   = 0;

        QGauss<dim> quadrature_formula(degree)
    }











}











//