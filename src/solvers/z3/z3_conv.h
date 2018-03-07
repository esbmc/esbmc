/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef _ESBMC_SOLVERS_Z3_Z3_CONV_H
#define _ESBMC_SOLVERS_Z3_Z3_CONV_H

#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <solvers/prop/pointer_logic.h>
#include <solvers/smt/smt_conv.h>
#include <util/hash_cont.h>
#include <util/irep2.h>
#include <util/namespace.h>
#include <vector>
#include <z3pp.h>

class z3_smt_ast : public solver_smt_ast<z3::expr>
{
public:
  using solver_smt_ast<z3::expr>::solver_smt_ast;
  ~z3_smt_ast() override = default;

  const smt_ast *update(
    smt_convt *ctx,
    const smt_ast *value,
    unsigned int idx,
    expr2tc idx_expr) const override;

  const smt_ast *project(smt_convt *ctx, unsigned int elem) const override;

  void dump() const override;
};

class z3_convt : public smt_convt,
                 public tuple_iface,
                 public array_iface,
                 public fp_convt
{
public:
  z3_convt(bool int_encoding, const namespacet &ns);
  ~z3_convt() override;

private:
  void intr_push_ctx();
  void intr_pop_ctx();

public:
  void push_ctx() override;
  void pop_ctx() override;
  smt_convt::resultt dec_solve() override;

  bool get_bool(const smt_ast *a) override;
  BigInt get_bv(smt_astt a) override;
  ieee_floatt get_fpbv(smt_astt a) override;
  expr2tc get_array_elem(
    const smt_ast *array,
    uint64_t index,
    const type2tc &subtype) override;

  z3::expr
  mk_tuple_update(const z3::expr &t, unsigned i, const z3::expr &new_val);
  z3::expr mk_tuple_select(const z3::expr &t, unsigned i);

  // SMT-abstraction migration:
  smt_astt mk_func_app(
    const smt_sort *s,
    smt_func_kind k,
    const smt_ast *const *args,
    unsigned int numargs) override;

  smt_sortt mk_bool_sort() override;
  smt_sortt mk_real_sort() override;
  smt_sortt mk_int_sort() override;
  smt_sortt mk_bv_sort(const smt_sort_kind k, std::size_t width) override;
  smt_sortt mk_array_sort(smt_sortt domain, smt_sortt range) override;
  smt_sortt mk_bv_fp_sort(std::size_t ew, std::size_t sw) override;
  smt_sortt mk_bv_fp_rm_sort() override;
  smt_sortt mk_fpbv_sort(const unsigned ew, const unsigned sw) override;
  smt_sortt mk_fpbv_rm_sort() override;

  smt_astt mk_smt_int(const mp_integer &theint, bool sign) override;
  smt_astt mk_smt_real(const std::string &str) override;
  smt_astt mk_smt_bv(smt_sortt s, const mp_integer &theint) override;
  smt_astt mk_smt_fpbv(const ieee_floatt &thereal) override;
  smt_astt mk_smt_fpbv_nan(unsigned ew, unsigned sw) override;
  smt_astt mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw) override;
  smt_astt mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm) override;

  smt_astt
  mk_smt_fpbv_fma(expr2tc v1, expr2tc v2, expr2tc v3, expr2tc rm) override;
  smt_astt
  mk_smt_typecast_from_fpbv_to_ubv(expr2tc from, std::size_t width) override;
  smt_astt
  mk_smt_typecast_from_fpbv_to_sbv(expr2tc from, std::size_t width) override;
  smt_astt mk_smt_typecast_from_fpbv_to_fpbv(
    expr2tc from,
    type2tc to,
    expr2tc rm) override;
  smt_astt
  mk_smt_typecast_ubv_to_fpbv(expr2tc from, type2tc to, expr2tc rm) override;
  smt_astt
  mk_smt_typecast_sbv_to_fpbv(expr2tc from, type2tc to, expr2tc rm) override;
  smt_astt mk_smt_fpbv_add(expr2tc lhs, expr2tc rhs, expr2tc rm) override;
  smt_astt mk_smt_fpbv_sub(expr2tc lhs, expr2tc rhs, expr2tc rm) override;
  smt_astt mk_smt_fpbv_mul(expr2tc lhs, expr2tc rhs, expr2tc rm) override;
  smt_astt mk_smt_fpbv_div(expr2tc lhs, expr2tc rhs, expr2tc rm) override;
  smt_astt mk_smt_nearbyint_from_float(expr2tc from, expr2tc rm) override;
  smt_astt mk_smt_fpbv_sqrt(expr2tc rd, expr2tc rm) override;

  smt_astt mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs) override;
  smt_astt mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs) override;
  smt_astt mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs) override;
  smt_astt mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs) override;
  smt_astt mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs) override;
  smt_astt mk_smt_fpbv_is_nan(smt_astt op) override;
  smt_astt mk_smt_fpbv_is_inf(smt_astt op) override;
  smt_astt mk_smt_fpbv_is_normal(smt_astt op) override;
  smt_astt mk_smt_fpbv_is_zero(smt_astt op) override;
  smt_astt mk_smt_fpbv_is_negative(smt_astt op) override;
  smt_astt mk_smt_fpbv_is_positive(smt_astt op) override;
  smt_astt mk_smt_fpbv_abs(smt_astt op) override;
  smt_astt mk_smt_fpbv_neg(smt_astt op) override;

  smt_astt mk_smt_bool(bool val) override;
  smt_astt mk_array_symbol(
    const std::string &name,
    const smt_sort *s,
    smt_sortt array_subtype) override;
  smt_astt mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  smt_sortt mk_struct_sort(const type2tc &type) override;
  smt_astt
  mk_extract(const smt_ast *a, unsigned int high, unsigned int low) override;
  smt_astt mk_sign_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_zero_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_concat(smt_astt a, smt_astt b) override;
  smt_astt mk_ite(smt_astt cond, smt_astt t, smt_astt f) override;

  const smt_ast *make_disjunct(const ast_vec &v) override;
  const smt_ast *make_conjunct(const ast_vec &v) override;

  smt_astt tuple_create(const expr2tc &structdef) override;
  smt_astt tuple_fresh(const smt_sort *s, std::string name = "") override;
  expr2tc tuple_get(const expr2tc &expr) override;

  const smt_ast *tuple_array_create(
    const type2tc &array_type,
    const smt_ast **input_args,
    bool const_array,
    const smt_sort *domain) override;

  smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s) override;
  smt_astt mk_tuple_array_symbol(const expr2tc &expr) override;
  smt_astt tuple_array_of(const expr2tc &init, unsigned long domain_width)
    override;

  const smt_ast *convert_array_of(smt_astt init_val, unsigned long domain_width)
    override;

  void add_array_constraints_for_solving() override;
  void add_tuple_constraints_for_solving() override;
  void push_array_ctx() override;
  void pop_array_ctx() override;
  void push_tuple_ctx() override;
  void pop_tuple_ctx() override;

  void assert_ast(const smt_ast *a) override;

  const std::string solver_text() override
  {
    unsigned int major, minor, build, revision;
    Z3_get_version(&major, &minor, &build, &revision);
    std::stringstream ss;
    ss << "Z3 v" << major << "." << minor << "." << build;
    return ss.str();
  }

  void dump_smt() override;
  void print_model() override;

  // Some useful types
public:
  inline z3_smt_ast *new_ast(z3::expr _e, const smt_sort *_s)
  {
    return new z3_smt_ast(this, _s, _e);
  }

  //  Must be first member; that way it's the last to be destroyed.
  z3::context z3_ctx;
  z3::solver solver;
  z3::model model;

  std::list<z3::expr> assumpt;
  std::list<std::list<z3::expr>::iterator> assumpt_ctx_stack;
};

#endif /* _ESBMC_SOLVERS_Z3_Z3_CONV_H_ */
