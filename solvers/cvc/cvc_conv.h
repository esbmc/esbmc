#ifndef _ESBMC_SOLVERS_CVC_CVC_CONV_H_
#define _ESBMC_SOLVERS_CVC_CVC_CONV_H_

#include <solvers/smt/smt_conv.h>

#include <cvc4/cvc4.h>

class cvc_smt_sort : public smt_sort
{
public:
#define cvc_sort_downcast(x) static_cast<const cvc_smt_sort *>(x)
  cvc_smt_sort(smt_sort_kind i, CVC4::Type &_t) : smt_sort(i), t(_t) { }
  virtual ~cvc_smt_sort() { }
  virtual unsigned long get_domain_width(void) const {
    return array_dom_width;
  }
  CVC4::Type t;
  unsigned int array_dom_width;
};

class cvc_smt_ast : public smt_ast
{
public:
#define cvc_ast_downcast(x) static_cast<const cvc_smt_ast *>(x)
  cvc_smt_ast(const smt_sort *_s, CVC4::Expr &_e) : smt_ast(_s), e(_e) { }
  virtual ~cvc_smt_ast() { }

  CVC4::Expr e;
};

class cvc_convt : public smt_convt
{
public:
  cvc_convt(bool is_cpp, bool int_encoding, const namespacet &ns);
  ~cvc_convt();

  virtual resultt dec_solve();
  virtual tvt l_get(const smt_ast *l);
  virtual const std::string solver_text();

  virtual void assert_ast(const smt_ast *a);

  virtual smt_ast *mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast * const *args,
                               unsigned int numargs);
  virtual smt_sort *mk_sort(const smt_sort_kind k, ...);
  virtual smt_ast *mk_smt_int(const mp_integer &theint, bool sign);
  virtual smt_ast *mk_smt_real(const std::string &str);
  virtual smt_ast *mk_smt_bvint(const mp_integer &theint, bool sign,
                                unsigned int w);
  virtual smt_ast *mk_smt_bool(bool val);
  virtual smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s);
  virtual smt_sort *mk_struct_sort(const type2tc &type);
  virtual smt_sort *mk_union_sort(const type2tc &type);
  virtual smt_ast *mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s);

  expr2tc get_bool(const smt_ast *a);
  expr2tc get_bv(const type2tc &t, const smt_ast *a);
  expr2tc get_array_elem(const smt_ast *array, uint64_t index,
                         const smt_sort *sort);

  CVC4::ExprManager em;
  CVC4::SmtEngine smt;
  CVC4::SymbolTable sym_tab;
};

#endif /* _ESBMC_SOLVERS_CVC_CVC_CONV_H_ */
