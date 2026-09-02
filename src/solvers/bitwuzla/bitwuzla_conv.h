#ifndef _ESBMC_SOLVERS_BITWUZLA_BITWUZLA_CONV_H_
#define _ESBMC_SOLVERS_BITWUZLA_BITWUZLA_CONV_H_

#include <map>
#include <memory>
#include <tuple>
#include <solvers/smt/smt_solver.h>
#include <irep2/irep2.h>
#include <util/symtab/namespace.h>

#include <bitwuzla/cpp/bitwuzla.h>

class bitw_smt_ast : public solver_smt_ast<bitwuzla::Term>
{
public:
  using solver_smt_ast<bitwuzla::Term>::solver_smt_ast;
  ~bitw_smt_ast() override = default;

  smt_astt with_sort(smt_solver_baset *ctx, smt_sortt s) const override;
  void dump() const override;
};

class bitwuzla_convt : public smt_solver_baset,
                       public array_iface,
                       public fp_convt
{
public:
  bitwuzla_convt(const namespacet &ns, const optionst &options);
  ~bitwuzla_convt() override;

  void push_ctx() override;
  void pop_ctx() override;
  smt_resultt dec_solve() override;
  const std::string solver_text() override;

  void assert_ast(smt_astt a) override;

  smt_astt mk_bvadd(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsub(smt_astt a, smt_astt b) override;
  smt_astt mk_bvmul(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsmod(smt_astt a, smt_astt b) override;
  smt_astt mk_bvumod(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsdiv(smt_astt a, smt_astt b) override;
  smt_astt mk_bvudiv(smt_astt a, smt_astt b) override;
  smt_astt mk_bvshl(smt_astt a, smt_astt b) override;
  smt_astt mk_bvashr(smt_astt a, smt_astt b) override;
  smt_astt mk_bvlshr(smt_astt a, smt_astt b) override;
  smt_astt mk_bvneg(smt_astt a) override;
  smt_astt mk_bvnot(smt_astt a) override;
  smt_astt mk_bvxor(smt_astt a, smt_astt b) override;
  smt_astt mk_bvor(smt_astt a, smt_astt b) override;
  smt_astt mk_bvand(smt_astt a, smt_astt b) override;
  smt_astt mk_implies(smt_astt a, smt_astt b) override;
  smt_astt mk_xor(smt_astt a, smt_astt b) override;
  smt_astt mk_or(smt_astt a, smt_astt b) override;
  smt_astt mk_and(smt_astt a, smt_astt b) override;
  smt_astt mk_not(smt_astt a) override;
  smt_astt mk_bvult(smt_astt a, smt_astt b) override;
  smt_astt mk_bvslt(smt_astt a, smt_astt b) override;
  smt_astt mk_bvugt(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsgt(smt_astt a, smt_astt b) override;
  smt_astt mk_bvule(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsle(smt_astt a, smt_astt b) override;
  smt_astt mk_bvuge(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsge(smt_astt a, smt_astt b) override;
  smt_astt mk_eq(smt_astt a, smt_astt b) override;
  smt_astt mk_neq(smt_astt a, smt_astt b) override;
  smt_astt mk_store(smt_astt a, smt_astt b, smt_astt c) override;
  smt_astt mk_select(smt_astt a, smt_astt b) override;

  smt_sortt mk_bool_sort() override;
  smt_sortt mk_bv_sort(std::size_t width) override;
  smt_sortt mk_array_sort(smt_sortt domain, smt_sortt range) override;
  smt_sortt mk_fbv_sort(std::size_t width) override;
  smt_sortt mk_bvfp_sort(std::size_t width, std::size_t swidth) override;
  smt_sortt mk_bvfp_rm_sort() override;

  /* Native floating-point support, the default; under --fp2bv these are never
   * reached because solve.cpp installs a plain fp_convt instead. */
  smt_sortt mk_fpbv_sort(const unsigned ew, const unsigned sw) override;
  smt_sortt mk_fpbv_rm_sort() override;

  smt_astt mk_smt_fpbv(const ieee_floatt &thereal) override;
  smt_astt mk_smt_fpbv_nan(bool sgn, unsigned ew, unsigned sw) override;
  smt_astt mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw) override;
  smt_astt mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm) override;

  smt_astt mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm) override;
  smt_astt mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm) override;
  smt_astt mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm) override;
  smt_astt mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm) override;
  smt_astt mk_smt_fpbv_rem(smt_astt lhs, smt_astt rhs) override;
  smt_astt
  mk_smt_fpbv_fma(smt_astt v1, smt_astt v2, smt_astt v3, smt_astt rm) override;
  smt_astt mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm) override;
  smt_astt mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm) override;

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

  smt_astt
  mk_smt_typecast_from_fpbv_to_ubv(smt_astt from, std::size_t width) override;
  smt_astt
  mk_smt_typecast_from_fpbv_to_sbv(smt_astt from, std::size_t width) override;
  smt_astt mk_smt_typecast_from_fpbv_to_fpbv(
    smt_astt from,
    smt_sortt to,
    smt_astt rm) override;
  smt_astt mk_smt_typecast_ubv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
    override;
  smt_astt mk_smt_typecast_sbv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
    override;

  smt_astt mk_from_bv_to_fp(smt_astt op, smt_sortt to) override;
  smt_astt mk_from_fp_to_bv(smt_astt op) override;
  ieee_floatt get_fpbv(smt_astt a) override;

  smt_astt mk_smt_int(const BigInt &theint) override;
  smt_astt mk_smt_real(const std::string &str) override;
  smt_astt mk_smt_bv(const BigInt &theint, smt_sortt s) override;
  smt_astt mk_smt_bool(bool val) override;
  smt_astt mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  smt_astt mk_smt_uninterpreted_function(
    const std::string &name,
    const std::vector<smt_astt> &args,
    smt_sortt rangesort) override;
  smt_astt mk_array_symbol(
    const std::string &name,
    const smt_sort *s,
    smt_sortt array_subtype) override;
  smt_astt mk_extract(smt_astt a, unsigned int high, unsigned int low) override;
  smt_astt mk_sign_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_zero_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_concat(smt_astt a, smt_astt b) override;
  smt_astt mk_ite(smt_astt cond, smt_astt t, smt_astt f) override;

  smt_astt
  convert_array_of(smt_astt init_val, unsigned long domain_width) override;

  tvt get_bool(smt_astt a) override;
  BigInt get_bv(smt_astt a, bool is_signed) override;
  expr2tc get_array_elem(smt_astt array, uint64_t index, const type2tc &subtype)
    override;

  smt_astt overflow_arith(const expr2tc &expr) override;

  std::string dump_smt() override;
  void print_model() override;

  smt_astt mk_quantifier(
    bool is_forall,
    std::vector<smt_astt> lhs,
    smt_astt rhs) override;

  /* Declaration order is the destruction contract: bitw is destroyed before
   * the term manager and options it was built from. */
  bitwuzla::TermManager tm;
  bitwuzla::Options bitw_options;
  std::unique_ptr<bitwuzla::Bitwuzla> bitw;

  symtabt symtable;

  /** Uninterpreted-function declarations, keyed by name. Bitwuzla mints a fresh
   *  constant on each mk_const, so the function term is cached here
   * and reused across applications, giving native functional congruence. */
  std::unordered_map<std::string, bitwuzla::Term> uf_decls;

private:
  /** Identifies a sort by kind plus the two widths that parameterise it: the
   *  bit-width for bit-vectors, (exponent, significand) for floating-point, and
   *  (domain width, range sort) for arrays. */
  typedef std::tuple<smt_sort_kind, uint64_t, uint64_t> sort_keyt;

  /** Sorts are immutable and outlive every context, so one instance per
   *  distinct sort suffices. mk_extract, mk_concat and the extends ask for a
   *  bit-vector sort per call, and neither the solver_smt_sort nor the
   *  Bitwuzla sort reference behind it is ever freed. */
  std::map<sort_keyt, smt_sortt> bitw_sorts;

  template <typename buildt>
  smt_sortt cached_sort(const sort_keyt &key, buildt build)
  {
    auto it = bitw_sorts.find(key);
    if (it == bitw_sorts.end())
      it = bitw_sorts.emplace(key, build()).first;
    return it->second;
  }

  smt_astt
  mk_fp_arith(bitwuzla::Kind kind, smt_astt lhs, smt_astt rhs, smt_astt rm);
  smt_astt mk_fp_pred(bitwuzla::Kind kind, smt_astt lhs, smt_astt rhs);
  smt_astt mk_fp_class(bitwuzla::Kind kind, smt_astt op);

  /** Bitwuzla has no fp.to_ieee_bv, so the bit pattern of an FP term is
   *  reached through a fresh bit-vector symbol b constrained by
   *  term = ((_ to_fp e s) b), as the cvc4/cvc5 backends do. Counter for
   *  those symbols' names. */
  unsigned to_bv_counter = 0;
};

#endif /* _ESBMC_SOLVERS_BITWUZLA_BITWUZLA_CONV_H_ */
