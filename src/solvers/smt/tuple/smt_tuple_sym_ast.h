#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_SYM_AST_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_SYM_AST_H_

#include <solvers/smt/smt_conv.h>

class tuple_sym_smt_ast;
typedef const tuple_sym_smt_ast *tuple_sym_smt_astt;

class smt_tuple_node_flattener;

class tuple_sym_smt_ast : public smt_ast
{
public:
  /** Primary constructor.
   *  @param s The sort of the tuple, of type tuple_smt_sort.
   *  @param _name The symbol prefix of the variables representing this tuples
   *               value. */
  tuple_sym_smt_ast(smt_convt *ctx, smt_sortt s, std::string _name)
    : smt_ast(ctx, s), name(std::move(_name))
  {
  }
  ~tuple_sym_smt_ast() override = default;

  /** The symbol prefix of the variables representing this tuples value, as a
   *  string (i.e., no associated type). */
  const std::string name;

  smt_astt ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const override;
  smt_astt eq(smt_convt *ctx, smt_astt other) const override;
  smt_astt update(
    smt_convt *ctx,
    smt_astt value,
    unsigned int idx,
    expr2tc idx_expr = expr2tc()) const override;
  smt_astt select(smt_convt *ctx, const expr2tc &idx) const override;
  smt_astt project(smt_convt *ctx, unsigned int elem) const override;

  void dump() const override
  {
  }
};

inline tuple_sym_smt_astt to_tuple_sym_ast(smt_astt a)
{
  tuple_sym_smt_astt ta = dynamic_cast<tuple_sym_smt_astt>(a);
  assert(ta != nullptr && "Tuple AST mismatch");
  return ta;
}

#endif
