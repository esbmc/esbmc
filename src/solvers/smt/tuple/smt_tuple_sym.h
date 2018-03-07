#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_SYM_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_SYM_H_

#include <solvers/smt/smt_conv.h>
#include <util/namespace.h>

class tuple_sym_smt_ast;
typedef const tuple_sym_smt_ast *tuple_sym_smt_astt;

class smt_tuple_sym_flattener : public tuple_iface
{
public:
  smt_tuple_sym_flattener(smt_convt *_ctx, const namespacet &_ns)
    : ctx(_ctx), ns(_ns)
  {
  }

  virtual ~smt_tuple_sym_flattener() = default;

  smt_sortt mk_struct_sort(const type2tc &type) override;
  smt_astt tuple_create(const expr2tc &structdef) override;
  smt_astt tuple_fresh(smt_sortt s, std::string name = "") override;
  smt_astt tuple_array_of(const expr2tc &init_value, unsigned long domain_width)
    override;
  smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s) override;
  smt_astt mk_tuple_array_symbol(const expr2tc &expr) override;
  expr2tc tuple_get(const expr2tc &expr) override;

  expr2tc tuple_get_rec(tuple_sym_smt_astt tuple);
  smt_astt tuple_array_create(
    const type2tc &array_type,
    smt_astt *input_args,
    bool const_array,
    smt_sortt domain) override;

  void add_tuple_constraints_for_solving() override;
  void push_tuple_ctx() override;
  void pop_tuple_ctx() override;

  smt_convt *ctx;
  const namespacet &ns;
};

#endif
