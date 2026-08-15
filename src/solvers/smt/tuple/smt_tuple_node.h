#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_NODE_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_NODE_H_

#include <solvers/smt/array_conv.h>
#include <solvers/smt/smt_solver.h>
#include <util/symtab/namespace.h>

class tuple_node_smt_ast;
typedef const tuple_node_smt_ast *tuple_node_smt_astt;

class smt_tuple_node_flattener : public tuple_iface
{
public:
  smt_tuple_node_flattener(smt_solver_baset *_ctx, const namespacet &_ns)
    : ctx(_ctx), ns(_ns), array_conv(_ctx)
  {
  }

  virtual ~smt_tuple_node_flattener() = default;

  smt_sortt mk_struct_sort(const type2tc &type) override;
  smt_astt tuple_create(const expr2tc &structdef) override;
  smt_astt tuple_fresh(smt_sortt s, std::string name = "") override;
  smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s) override;
  expr2tc tuple_get(const expr2tc &expr) override;
  expr2tc tuple_get(const type2tc &type, smt_astt a) override;

  expr2tc tuple_get_rec(tuple_node_smt_astt tuple);

  expr2tc tuple_get_array_elem(
    smt_astt array,
    uint64_t index,
    const type2tc &subtype) override;

  smt_astt mk_tuple_array_symbol(const expr2tc &expr) override;
  smt_astt tuple_array_of(const expr2tc &init_value, unsigned long domain_width)
    override;
  smt_astt tuple_array_create(
    const type2tc &array_type,
    smt_astt *input_args,
    bool const_array,
    smt_sortt domain) override;

  void add_tuple_constraints_for_solving() override;
  void push_tuple_ctx() override;
  void pop_tuple_ctx() override;

  /** Record that @p tuple had its elements installed at the current context
   *  level, having itself been created at a shallower one — so pop_tuple_ctx
   *  must clear them before the ASTs they name are destroyed. Keying on the
   *  level the vector was *installed* at rather than the one its contents were
   *  allocated at is what makes the clear both safe and exact: installation
   *  never precedes allocation, pops descend, and the vector a tuple held
   *  before the install was empty. */
  void note_elements_populated(tuple_node_smt_ast *tuple);

  smt_solver_baset *ctx;
  const namespacet &ns;
  array_convt array_conv;

private:
  /** Tuples needing their elements cleared, keyed by the level the elements
   *  were populated at. */
  std::map<unsigned int, std::vector<tuple_node_smt_ast *>> populated_elements;
};

#endif
