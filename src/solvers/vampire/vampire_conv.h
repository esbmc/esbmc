/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef _ESBMC_SOLVERS_vampire_vampire_CONV_H
#define _ESBMC_SOLVERS_vampire_vampire_CONV_H

#include <solvers/smt/smt_conv.h>
#include <Solver.hpp>

class vampire_smt_ast : public solver_smt_ast<Vampire::Expression>
{
public:
  using solver_smt_ast<Vampire::Expression>::solver_smt_ast;
  ~vampire_smt_ast() override = default;

  /*smt_astt
  update(smt_convt *ctx, smt_astt value, unsigned int idx, expr2tc idx_expr)
    const override;

  smt_astt project(smt_convt *ctx, unsigned int elem) const override;*/

  void dump() const override;
};

class vampire_convt : public smt_convt,
                 //public tuple_iface,
                 public array_iface,
                 public fp_convt
{
public:
  vampire_convt(bool int_encoding, const namespacet &ns);
  ~vampire_convt() override = default;

public:
  void push_ctx() override;
  void pop_ctx() override;
  smt_convt::resultt dec_solve() override;

  bool get_bool(smt_astt a) override;
  BigInt get_bv(smt_astt a, bool is_signed) override;

  expr2tc get_array_elem(smt_astt array, uint64_t index, const type2tc &subtype)
    override;

  Vampire::Expression
  mk_tuple_update(const Vampire::Expression &t, unsigned i, const Vampire::Expression &new_val);
  Vampire::Expression mk_tuple_select(const Vampire::Expression &t, unsigned i);

  // SMT-abstraction migration:
  smt_astt mk_add(smt_astt a, smt_astt b) override;
  smt_astt mk_sub(smt_astt a, smt_astt b) override;
  smt_astt mk_mul(smt_astt a, smt_astt b) override;
  smt_astt mk_mod(smt_astt a, smt_astt b) override;
  smt_astt mk_div(smt_astt a, smt_astt b) override;
  smt_astt mk_shl(smt_astt a, smt_astt b) override;
  smt_astt mk_neg(smt_astt a) override;
  smt_astt mk_implies(smt_astt a, smt_astt b) override;
  smt_astt mk_xor(smt_astt a, smt_astt b) override;
  smt_astt mk_or(smt_astt a, smt_astt b) override;
  smt_astt mk_and(smt_astt a, smt_astt b) override;
  smt_astt mk_not(smt_astt a) override;
  smt_astt mk_lt(smt_astt a, smt_astt b) override;
  smt_astt mk_gt(smt_astt a, smt_astt b) override;
  smt_astt mk_le(smt_astt a, smt_astt b) override;
  smt_astt mk_ge(smt_astt a, smt_astt b) override;
  smt_astt mk_eq(smt_astt a, smt_astt b) override;
  smt_astt mk_neq(smt_astt a, smt_astt b) override;
  smt_astt mk_store(smt_astt a, smt_astt b, smt_astt c) override;
  smt_astt mk_select(smt_astt a, smt_astt b) override;
  smt_astt mk_real2int(smt_astt a) override;
  smt_astt mk_int2real(smt_astt a) override;
  smt_astt mk_isint(smt_astt a) override;

  smt_sortt mk_bool_sort() override;
  smt_sortt mk_real_sort() override;
  smt_sortt mk_int_sort() override;
  smt_sortt mk_array_sort(smt_sortt domain, smt_sortt range) override;

  smt_astt mk_smt_int(const BigInt &theint) override;
  smt_astt mk_smt_real(const std::string &str) override;
  smt_astt mk_smt_bv(const BigInt &theint, smt_sortt s) override;

  smt_astt mk_smt_bool(bool val) override;
  smt_astt mk_array_symbol(
    const std::string &name,
    const smt_sort *s,
    smt_sortt array_subtype) override;
  smt_astt mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  //smt_sortt mk_struct_sort(const type2tc &type) override;
  smt_astt mk_extract(smt_astt a, unsigned int high, unsigned int low) override;
  smt_astt mk_sign_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_zero_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_concat(smt_astt a, smt_astt b) override;
  smt_astt mk_ite(smt_astt cond, smt_astt t, smt_astt f) override;

  /*smt_astt tuple_create(const expr2tc &structdef) override;
  smt_astt tuple_fresh(const smt_sort *s, std::string name = "") override;
  expr2tc tuple_get(const expr2tc &expr) override;

  smt_astt tuple_array_create(
    const type2tc &array_type,
    smt_astt *input_args,
    bool const_array,
    const smt_sort *domain) override;

  smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s) override;
  smt_astt mk_tuple_array_symbol(const expr2tc &expr) override;
  smt_astt
  tuple_array_of(const expr2tc &init, unsigned long domain_width) override;*/

  smt_astt
  convert_array_of(smt_astt init_val, unsigned long domain_width) override;

  void assert_ast(smt_astt a) override;

  const std::string solver_text() override
  {
    std::stringstream ss;
    ss << "Vampire v" << solver->version() << " commit: " << solver->commit();
    return ss.str();
  }

  void dump_smt() override;
  void print_model() override;

public:
  Vampire::Solver* solver;
};

#endif /* _ESBMC_SOLVERS_vampire_vampire_CONV_H_ */
