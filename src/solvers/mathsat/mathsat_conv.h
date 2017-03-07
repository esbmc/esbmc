#ifndef _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_
#define _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_

#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_tuple_flat.h>
#include <solvers/smt/fp_conv.h>

#include <mathsat.h>

class mathsat_smt_sort : public smt_sort
{
public:
#define mathsat_sort_downcast(x) static_cast<const mathsat_smt_sort *>(x)
  mathsat_smt_sort(smt_sort_kind i, msat_type _t) : smt_sort(i), t(_t), rangesort(NULL) { }
  mathsat_smt_sort(smt_sort_kind i, msat_type _t, size_t w)
    : smt_sort(i, w), t(_t), rangesort(NULL) { }
  mathsat_smt_sort(smt_sort_kind i, msat_type _t, size_t r_w, size_t dom_w)
    : smt_sort(i, r_w, dom_w), t(_t), rangesort(NULL) { }
  mathsat_smt_sort(smt_sort_kind i, msat_type _t, size_t w, size_t dw,
                   const smt_sort *_rangesort)
    : smt_sort(i, w, dw), t(_t), rangesort(_rangesort) {}
  virtual ~mathsat_smt_sort();

  msat_type t;
  const smt_sort *rangesort;
};

class mathsat_smt_ast : public smt_ast
{
public:
#define mathsat_ast_downcast(x) static_cast<const mathsat_smt_ast *>(x)
  mathsat_smt_ast(smt_convt *ctx, const smt_sort *_s, msat_term _t)
    : smt_ast(ctx, _s), t(_t) { }
  virtual ~mathsat_smt_ast();

  virtual const smt_ast *select(smt_convt *ctx, const expr2tc &idx) const;
  virtual void dump() const;

  msat_term t;
};

class mathsat_convt : public smt_convt, public array_iface, public fp_convt
{
public:
  mathsat_convt(bool int_encoding, const namespacet &ns);
  ~mathsat_convt(void);

  virtual resultt dec_solve();
  virtual const std::string solver_text();

  virtual void assert_ast(const smt_ast *a);

  virtual smt_ast *mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast * const *args,
                               unsigned int numargs);
  virtual smt_sortt mk_sort(const smt_sort_kind k, ...);
  virtual smt_ast *mk_smt_int(const mp_integer &theint, bool sign);
  virtual smt_ast *mk_smt_real(const std::string &str);
  virtual smt_ast *mk_smt_bvint(const mp_integer &theint, bool sign,
                                unsigned int w);
  virtual smt_astt mk_smt_fpbv(const ieee_floatt &thereal);
  virtual smt_astt mk_smt_fpbv_nan(unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm);
  virtual smt_astt mk_smt_typecast_from_fpbv(const typecast2t &cast);
  virtual smt_astt mk_smt_typecast_to_fpbv(const typecast2t &cast);
  virtual smt_astt mk_smt_nearbyint_from_float(const nearbyint2t &expr);
  virtual smt_astt mk_smt_fpbv_arith_ops(const expr2tc &expr);
  virtual smt_astt mk_smt_fpbv_fma(const expr2tc &expr);
  virtual smt_ast *mk_smt_bool(bool val);
  virtual smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s);
  virtual smt_ast *mk_array_symbol(const std::string &name, const smt_sort *s,
                                   smt_sortt array_subtype);
  virtual smt_sort *mk_struct_sort(const type2tc &type);
  virtual smt_sortt mk_fpbv_sort(const unsigned ew, const unsigned sw);
  virtual smt_ast *mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s);

  void push_ctx();
  void pop_ctx();

  expr2tc get_bool(const smt_ast *a);
  expr2tc get_bv(const type2tc &t, const smt_ast *a);
  virtual expr2tc get_fpbv(const type2tc &t, smt_astt a);
  expr2tc get_array_elem(
    const smt_ast *array,
    uint64_t idx,
    const type2tc &elem_sort);

  virtual const smt_ast *convert_array_of(smt_astt init_val,
                                          unsigned long domain_width);

  virtual void add_array_constraints_for_solving();
  void push_array_ctx(void);
  void pop_array_ctx(void);

  size_t get_exp_width(smt_sortt sort);
  size_t get_mant_width(smt_sortt sort);

  virtual void dump_SMT();

  // MathSAT data.
  msat_config cfg;
  msat_env env;
};

#endif /* _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_ */
