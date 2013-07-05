#ifndef _ESBMC_SOLVERS_SMTLIB_CONV_H_
#define _ESBMC_SOLVERS_SMTLIB_CONV_H_

#include <solvers/smt/smt_conv.h>

class minisat_convt : public smt_convt {
public:
  minisat_convt(bool int_encoding, const namespacet &_ns, bool is_cpp,
                const optionst &opts);
  ~minisat_convt();

  virtual resultt dec_solve();
  virtual expr2tc get(const expr2tc &expr);
  virtual const std::string solver_text();
  virtual tvt l_get(literalt l);
  virtual void assert_lit(const literalt &l);
  virtual smt_ast* mk_func_app(const smt_sort *ressort, smt_func_kind f,
                               const smt_ast* const* args, unsigned int num);
  virtual smt_sort* mk_sort(smt_sort_kind k, ...);
  virtual literalt mk_lit(const smt_ast *val);
  virtual smt_ast* mk_smt_int(const mp_integer &intval, bool sign);
  virtual smt_ast* mk_smt_real(const std::string &value);
  virtual smt_ast* mk_smt_bvint(const mp_integer &inval, bool sign,
                                unsigned int w);
  virtual smt_ast* mk_smt_bool(bool boolval);
  virtual smt_ast* mk_smt_symbol(const std::string &name, const smt_sort *sort);
  virtual smt_sort* mk_struct_sort(const type2tc &t);
  virtual smt_sort* mk_union_sort(const type2tc&t);
  virtual smt_ast* mk_extract(const smt_ast *src, unsigned int high,
                              unsigned int low, const smt_sort *s);
};

#endif /* _ESBMC_SOLVERS_SMTLIB_CONV_H_ */
