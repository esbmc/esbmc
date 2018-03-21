#ifndef SOLVERS_SMT_SMT_AST_H_
#define SOLVERS_SMT_SMT_AST_H_

#include <solvers/smt/smt_sort.h>
#include <util/irep2_expr.h>

class smt_convt;

/** Storage of an SMT function application.
 *  This class represents a single SMT function app, abstractly. Solver
 *  converter classes must extend this and add whatever fields are necessary
 *  to represent a function application in the solver they support. A converted
 *  expression becomes an SMT function application; that is then handed around
 *  the rest of the SMT conversion code as an smt_ast.
 *
 *  While an expression becomes an smt_ast, the inverse is not true, and a
 *  single expression may in fact become many smt_asts in various places. See
 *  smt_convt for more detail on how conversion occurs.
 *
 *  The function arguments, and the actual function application itself are all
 *  abstract and dealt with by the solver converter class. Only the sort needs
 *  to be available for us to make conversion decisions.
 *  @see smt_convt
 *  @see smt_sort
 */

class smt_ast;
typedef const smt_ast *smt_astt;

class smt_ast
{
public:
  /** The sort of this function application. */
  smt_sortt sort;

  /** The solver context */
  const smt_convt *context;

  smt_ast(smt_convt *ctx, smt_sortt s);
  virtual ~smt_ast() = default;

  // "this" is the true operand.
  virtual smt_astt ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const;

  /** Abstractly produce an equality. Does the right thing (TM) whether it's
   *  a normal piece of AST or a tuple / array.
   *  @param ctx SMT context to produce the equality in.
   *  @param other Piece of AST to compare 'this' with.
   *  @return Boolean typed AST representing an equality */
  virtual smt_astt eq(smt_convt *ctx, smt_astt other) const;

  /** Abstractly produce an assign. Defaults to being an equality, however
   *  for some special cases up to the backend, there may be optimisations made
   *  for array or tuple assigns, and so forth.
   *  @param ctx SMT context to do the assignment in.
   *  @param sym Symbol to assign to
   *  @return AST representing the assigned symbol */
  virtual void assign(smt_convt *ctx, smt_astt sym) const;

  /** Abstractly produce an "update", i.e. an array 'with' or tuple 'with'.
   *  @param ctx SMT context to make this update in.
   *  @param value Value to insert into the updated field
   *  @param idx Array index or tuple field
   *  @param idx_expr If an array, expression representing the index
   *  @return AST of this' type, representing the update */
  virtual smt_astt update(
    smt_convt *ctx,
    smt_astt value,
    unsigned int idx,
    expr2tc idx_expr = expr2tc()) const;

  /** Select a value from an array, for both normal arrays and tuple arrays.
   *  @param ctx SMT context to produce this in.
   *  @param idx Index to select the value from.
   *  @return AST of the array's range sort representing the selected item */
  virtual smt_astt select(smt_convt *ctx, const expr2tc &idx) const;

  /** Project a member from a structure, or an field-array from a struct array.
   *  @param ctx SMT context to produce this in.
   *  @param elem Struct index to project.
   *  @return AST representing the chosen element / element-array */
  virtual smt_astt project(smt_convt *ctx, unsigned int elem) const;

  virtual void dump() const
  {
    std::cout << "Chosen solver doesn't support printing the AST\n";
  }
};

template <typename solver_ast>
class solver_smt_ast : public smt_ast
{
public:
  solver_smt_ast(smt_convt *ctx, smt_sortt s, solver_ast _a)
    : smt_ast(ctx, s), a(_a)
  {
  }

  solver_ast a;
};

#ifdef NDEBUG
#define dynamic_cast static_cast
#endif
template <typename derived_class>
const derived_class *to_solver_smt_ast(smt_astt s)
{
  return dynamic_cast<const derived_class *>(s);
}
#ifdef dynamic_cast
#undef dynamic_cast
#endif

#endif /* SOLVERS_SMT_SMT_AST_H_ */
