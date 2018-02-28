#ifndef _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_
#define _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_

#include <mathsat.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/fp/fp_conv.h>

class mathsat_smt_ast : public solver_smt_ast<msat_term>
{
public:
  using solver_smt_ast<msat_term>::solver_smt_ast;
  ~mathsat_smt_ast() override = default;

  void dump() const override;
};

class mathsat_convt : public smt_convt, public array_iface, public fp_convt
{
public:
  mathsat_convt(bool int_encoding, const namespacet &ns);
  ~mathsat_convt() override;

  resultt dec_solve() override;
  const std::string solver_text() override;

  void assert_ast(const smt_ast *a) override;

  smt_ast *mk_func_app(
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
  smt_ast *mk_smt_real(const std::string &str) override;
  smt_ast *mk_smt_bool(bool val) override;
  smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  smt_ast *mk_array_symbol(
    const std::string &name,
    const smt_sort *s,
    smt_sortt array_subtype) override;
  smt_ast *mk_extract(
    const smt_ast *a,
    unsigned int high,
    unsigned int low,
    const smt_sort *s) override;
  smt_astt mk_smt_bv(smt_sortt s, const mp_integer &theint) override;
  smt_astt mk_smt_fpbv(const ieee_floatt &thereal) override;
  smt_astt mk_smt_fpbv_nan(unsigned ew, unsigned sw) override;
  smt_astt mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw) override;
  smt_astt mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm) override;

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
  smt_astt mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs) override;
  smt_astt mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs) override;
  smt_astt mk_smt_fpbv_is_nan(smt_astt op) override;
  smt_astt mk_smt_fpbv_is_inf(smt_astt op) override;
  smt_astt mk_smt_fpbv_is_normal(smt_astt op) override;
  smt_astt mk_smt_fpbv_is_zero(smt_astt op) override;
  smt_astt mk_smt_fpbv_is_negative(smt_astt op) override;
  smt_astt mk_smt_fpbv_is_positive(smt_astt op) override;
  smt_astt mk_smt_fpbv_abs(smt_astt op) override;

  void push_ctx() override;
  void pop_ctx() override;

  bool get_bool(const smt_ast *a) override;
  BigInt get_bv(smt_astt a) override;
  ieee_floatt get_fpbv(smt_astt a) override;
  expr2tc get_array_elem(
    const smt_ast *array,
    uint64_t index,
    const type2tc &subtype) override;

  const smt_ast *convert_array_of(smt_astt init_val, unsigned long domain_width)
    override;

  void add_array_constraints_for_solving() override;
  void push_array_ctx() override;
  void pop_array_ctx() override;

  void check_msat_error(msat_term &r);

  void dump_smt() override;
  void print_model() override;

  // MathSAT data.
  msat_config cfg;
  msat_env env;
};

#endif /* _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_ */
