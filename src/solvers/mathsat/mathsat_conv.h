#ifndef _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_
#define _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_

#include <mathsat.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_tuple_flat.h>
#include <solvers/smt/fp_conv.h>

class mathsat_smt_sort : public smt_sort
{
public:
#define mathsat_sort_downcast(x) static_cast<const mathsat_smt_sort *>(x)
  mathsat_smt_sort(smt_sort_kind i, msat_type _s)
    : smt_sort(i), s(_s), rangesort(nullptr) { }

  mathsat_smt_sort(smt_sort_kind i, msat_type _s, size_t w)
    : smt_sort(i, w), s(_s), rangesort(nullptr) { }

  mathsat_smt_sort(smt_sort_kind i, msat_type _s, size_t w, size_t sw)
    : smt_sort(i, w, sw), s(_s), rangesort(nullptr) { }

  mathsat_smt_sort(smt_sort_kind i, msat_type _s, size_t w, size_t dw,
                   const smt_sort *_rangesort)
    : smt_sort(i, w, dw), s(_s), rangesort(_rangesort) { }

  ~mathsat_smt_sort() override = default;

  msat_type s;
  const smt_sort *rangesort;
};


class mathsat_smt_ast : public smt_ast
{
public:
#define mathsat_ast_downcast(x) static_cast<const mathsat_smt_ast *>(x)
  mathsat_smt_ast(smt_convt *ctx, const smt_sort *_s, msat_term _t)
    : smt_ast(ctx, _s), t(_t) { }
  ~mathsat_smt_ast() override = default;

  const smt_ast *select(smt_convt *ctx, const expr2tc &idx) const override;
  void dump() const override;

  msat_term t;
};

class mathsat_convt : public smt_convt, public array_iface, public fp_convt
{
public:
  mathsat_convt(bool int_encoding, const namespacet &ns);
  ~mathsat_convt() override;

  resultt dec_solve() override;
  const std::string solver_text() override;

  void assert_ast(const smt_ast *a) override;

  smt_ast *mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast * const *args,
                               unsigned int numargs) override;
  smt_sortt mk_sort(const smt_sort_kind k, ...) override;
  smt_astt mk_smt_int(const mp_integer &theint, bool sign) override;
  smt_ast *mk_smt_real(const std::string &str) override;
  smt_ast *mk_smt_bool(bool val) override;
  smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  smt_ast *mk_array_symbol(
    const std::string &name,
    const smt_sort *s,
    smt_sortt array_subtype) override;
  virtual smt_sort *mk_struct_sort(const type2tc &type);
  smt_ast *mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s) override;
  smt_astt mk_smt_bvint(
    const mp_integer &theint,
    bool sign,
    unsigned int w) override;

  smt_sortt mk_fpbv_sort(const unsigned ew, const unsigned sw) override;
  smt_astt mk_smt_fpbv(const ieee_floatt &thereal) override;
  smt_astt mk_smt_fpbv_nan(unsigned ew, unsigned sw) override;
  smt_astt mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw) override;
  smt_astt mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm) override;
  smt_astt mk_smt_typecast_from_fpbv(const typecast2t &cast) override;
  smt_astt mk_smt_typecast_to_fpbv(const typecast2t &cast) override;
  smt_astt mk_smt_nearbyint_from_float(const nearbyint2t &expr) override;
  smt_astt mk_smt_fpbv_arith_ops(const expr2tc &expr) override;
  smt_astt mk_smt_fpbv_fma(const expr2tc &expr) override;

  void push_ctx() override;
  void pop_ctx() override;

  expr2tc get_bool(const smt_ast *a) override;
  expr2tc get_bv(const type2tc &type, smt_astt a) override;
  expr2tc get_fpbv(const type2tc &t, smt_astt a) override;
  expr2tc get_array_elem(
    const smt_ast *array,
    uint64_t index,
    const type2tc &subtype) override;

  const smt_ast *convert_array_of(
    smt_astt init_val,
    unsigned long domain_width) override;

  void add_array_constraints_for_solving() override;
  void push_array_ctx() override;
  void pop_array_ctx() override;

  void dump_smt() override;

  // MathSAT data.
  msat_config cfg;
  msat_env env;
};

#endif /* _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_ */
