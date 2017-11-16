#ifndef _ESBMC_SOLVERS_CVC_CVC_CONV_H_
#define _ESBMC_SOLVERS_CVC_CVC_CONV_H_

#include <solvers/smt/smt_conv.h>
#include <cvc4/cvc4.h>

class cvc_smt_sort : public smt_sort
{
public:
#define cvc_sort_downcast(x) static_cast<const cvc_smt_sort *>(x)
  cvc_smt_sort(smt_sort_kind i, CVC4::Type _s)
    : smt_sort(i), s(_s), rangesort(nullptr) { }

  cvc_smt_sort(smt_sort_kind i, CVC4::Type _s, size_t w)
    : smt_sort(i, w), s(_s), rangesort(nullptr) { }

  cvc_smt_sort(smt_sort_kind i, CVC4::Type _s, size_t w, size_t sw)
    : smt_sort(i, w, sw), s(_s), rangesort(nullptr) { }

  cvc_smt_sort(smt_sort_kind i, CVC4::Type _s, size_t w, size_t dw,
              const smt_sort *_rangesort)
    : smt_sort(i, w, dw), s(_s), rangesort(_rangesort) { }

  ~cvc_smt_sort() override = default;

  CVC4::Type s;
  const smt_sort *rangesort;
};

class cvc_smt_ast : public smt_ast
{
public:
#define cvc_ast_downcast(x) static_cast<const cvc_smt_ast *>(x)
  cvc_smt_ast(smt_convt *ctx, const smt_sort *_s, CVC4::Expr &_e)
    : smt_ast(ctx, _s), e(_e) { }
  ~cvc_smt_ast() override = default;
  void dump() const override { abort(); }

  CVC4::Expr e;
};

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
    const smt_ast * const *args,
    unsigned int numargs) override;
  smt_sortt mk_sort(const smt_sort_kind k, ...) override;
  smt_ast *mk_smt_int(const mp_integer &theint, bool sign) override;
  smt_ast *mk_smt_real(const std::string &str) override;
  smt_ast *mk_smt_bvint(
    const mp_integer &theint,
    bool sign,
    unsigned int w) override;
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

  const smt_ast *convert_array_of(
    smt_astt init_val,
    unsigned long domain_width) override;

  void add_array_constraints_for_solving() override;
  void push_array_ctx() override;
  void pop_array_ctx() override;

  expr2tc get_bool(const smt_ast *a) override;
  expr2tc get_bv(const type2tc &type, smt_astt a) override;
  expr2tc get_array_elem(
    const smt_ast *array,
    uint64_t index,
    const type2tc &subtype) override;

  CVC4::ExprManager em;
  CVC4::SmtEngine smt;
  CVC4::SymbolTable sym_tab;
};

#endif /* _ESBMC_SOLVERS_CVC_CVC_CONV_H_ */
