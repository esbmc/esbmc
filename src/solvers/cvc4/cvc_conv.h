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

  smt_ast *mk_func_app(
    const smt_sort *s,
    smt_func_kind k,
    const smt_ast *const *args,
    unsigned int numargs) override;

  smt_sortt mk_bool_sort() override;
  smt_sortt mk_bv_sort(const smt_sort_kind k, std::size_t width) override;
  smt_sortt mk_array_sort(smt_sortt domain, smt_sortt range) override;
  smt_sortt mk_bv_fp_sort(std::size_t ew, std::size_t sw) override;
  smt_sortt mk_bv_fp_rm_sort() override;

  smt_ast *mk_smt_int(const mp_integer &theint, bool sign) override;
  smt_ast *mk_smt_real(const std::string &str) override;
  smt_ast *mk_smt_bv(smt_sortt s, const mp_integer &theint) override;
  smt_ast *mk_smt_bool(bool val) override;
  smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  smt_ast *mk_array_symbol(
    const std::string &name,
    const smt_sort *s,
    smt_sortt array_subtype) override;
  virtual smt_sort *mk_struct_sort(const type2tc &type);
  smt_ast *mk_extract(
    const smt_ast *a,
    unsigned int high,
    unsigned int low,
    const smt_sort *s) override;

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
