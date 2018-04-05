#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_NODE_AST_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_NODE_AST_H_

#include <solvers/smt/smt_conv.h>

class tuple_node_smt_ast;
typedef const tuple_node_smt_ast *tuple_node_smt_astt;

class smt_tuple_node_flattener;

/** Function app representing a tuple sorted value.
 *  This AST represents any kind of SMT function that results in something of
 *  a tuple sort. As documented in smt_tuple.c, the result of any kind of
 *  tuple operation that gets flattened is a symbol prefix, which is what this
 *  ast actually stores.
 *
 *  This AST should only be used in smt_tuple.c, if you're using it elsewhere
 *  think very hard about what you're trying to do. Its creation should also
 *  only occur if there is no tuple support in the solver being used, and a
 *  tuple creating method has been called.
 *
 *  @see smt_tuple.c */
class tuple_node_smt_ast : public smt_ast
{
public:
  /** Primary constructor.
   *  @param s The sort of the tuple, of type tuple_smt_sort.
   *  @param _name The symbol prefix of the variables representing this tuples
   *               value. */
  tuple_node_smt_ast(
    smt_tuple_node_flattener &f,
    smt_convt *ctx,
    smt_sortt s,
    std::string _name)
    : smt_ast(ctx, s), name(std::move(_name)), flat(f)
  {
  }
  ~tuple_node_smt_ast() override = default;

  /** The symbol prefix of the variables representing this tuples value, as a
   *  string (i.e., no associated type). */
  const std::string name;

  smt_tuple_node_flattener &flat;
  std::vector<smt_astt> elements;

  smt_astt ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const override;
  smt_astt eq(smt_convt *ctx, smt_astt other) const override;
  void assign(smt_convt *ctx, smt_astt sym) const override;
  smt_astt update(
    smt_convt *ctx,
    smt_astt value,
    unsigned int idx,
    expr2tc idx_expr = expr2tc()) const override;
  smt_astt select(smt_convt *ctx, const expr2tc &idx) const override;
  smt_astt project(smt_convt *ctx, unsigned int elem) const override;

  void dump() const override
  {
    std::cout << "name: " << name << '\n';
    for(auto const &e : elements)
      e->dump();
  }

  void make_free(smt_convt *ctx);
  void pre_ite(smt_convt *ctx, smt_astt cond, smt_astt falseop);
};

inline tuple_node_smt_astt to_tuple_node_ast(smt_astt a)
{
  tuple_node_smt_astt ta = dynamic_cast<tuple_node_smt_astt>(a);
  assert(ta != nullptr && "Tuple AST mismatch");
  return ta;
}

#endif
