#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_NODE_AST_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_NODE_AST_H_

#include <solvers/smt/smt_conv.h>
#include <solvers/smt/tuple/smt_tuple_sym_ast.h>

class array_sym_smt_ast;
typedef const array_sym_smt_ast *array_sym_smt_astt;

class array_sym_smt_ast : public tuple_sym_smt_ast
{
public:
  array_sym_smt_ast(smt_convt *ctx, smt_sortt s, const std::string &_name)
    : tuple_sym_smt_ast(ctx, s, _name)
  {
  }
  virtual ~array_sym_smt_ast() = default;

  smt_astt ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const override;
  smt_astt eq(smt_convt *ctx, smt_astt other) const override;
  smt_astt update(
    smt_convt *ctx,
    smt_astt value,
    unsigned int idx,
    expr2tc idx_expr = expr2tc()) const override;
  smt_astt select(smt_convt *ctx, const expr2tc &idx) const override;
  smt_astt project(smt_convt *ctx, unsigned int elem) const override;
  void assign(smt_convt *ctx, smt_astt sym) const override;
};

inline array_sym_smt_astt to_array_sym_ast(smt_astt a)
{
  array_sym_smt_astt ta = dynamic_cast<array_sym_smt_astt>(a);
  assert(ta != nullptr && "Tuple-Array AST mismatch");
  return ta;
}

#endif
