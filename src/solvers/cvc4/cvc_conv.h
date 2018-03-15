#ifndef _ESBMC_SOLVERS_CVC_CVC_CONV_H_
#define _ESBMC_SOLVERS_CVC_CVC_CONV_H_

#include <solvers/smt/smt_conv.h>
#include <cvc4/cvc4.h>

class cvc_convt : public smt_convt, public array_iface, public fp_convt
{
public:
  cvc_convt(bool int_encoding, const namespacet &ns);
  ~cvc_convt() override = default;

  resultt dec_solve() override;
  const std::string solver_text() override;

  void assert_ast(const smt_ast *a) override;

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
  smt_astt mk_bvnxor(smt_astt a, smt_astt b) override;
  smt_astt mk_bvnor(smt_astt a, smt_astt b) override;
  smt_astt mk_bvnand(smt_astt a, smt_astt b) override;
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
  smt_sortt mk_bvfp_sort(std::size_t ew, std::size_t sw) override;
  smt_sortt mk_bvfp_rm_sort() override;

  smt_astt mk_smt_int(const mp_integer &theint, bool sign) override;
  smt_astt mk_smt_real(const std::string &str) override;
  smt_astt mk_smt_bv(smt_sortt s, const mp_integer &theint) override;
  smt_astt mk_smt_bool(bool val) override;
  smt_astt mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  smt_astt mk_array_symbol(
    const std::string &name,
    const smt_sort *s,
    smt_sortt array_subtype) override;
  smt_astt
  mk_extract(const smt_ast *a, unsigned int high, unsigned int low) override;
  smt_astt mk_sign_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_zero_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_concat(smt_astt a, smt_astt b) override;
  smt_astt mk_ite(smt_astt cond, smt_astt t, smt_astt f) override;

  inline smt_astt new_ast(CVC4::Expr _e, const smt_sort *_s)
  {
    return new solver_smt_ast<CVC4::Expr>(this, _s, _e);
  }

  const smt_ast *
  convert_array_of(smt_astt init_val, unsigned long domain_width) override;

  void add_array_constraints_for_solving() override;
  void push_array_ctx() override;
  void pop_array_ctx() override;

  bool get_bool(const smt_ast *a) override;
  BigInt get_bv(smt_astt a) override;
  expr2tc get_array_elem(
    const smt_ast *array,
    uint64_t index,
    const type2tc &subtype) override;

  CVC4::ExprManager em;
  CVC4::SmtEngine smt;
  CVC4::SymbolTable sym_tab;
};

#endif /* _ESBMC_SOLVERS_CVC_CVC_CONV_H_ */
