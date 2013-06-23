#include <solvers/smt/smt_conv.h>

class metasmt_convt : public smt_convt
{
public:
  metasmt_convt(bool int_encoding, bool is_cpp, const namespacet &ns);
  virtual ~metasmt_convt();

  virtual void set_to(const expr2tc &expr, bool value);
  virtual resultt dec_solve();

  virtual expr2tc get(const expr2tc &expr);

  virtual tvt l_get(literalt a);
  virtual const std::string solver_text();


  virtual void assert_lit(const literalt &l);
  virtual smt_ast *mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast **args, unsigned int numargs);
  virtual smt_sort *mk_sort(const smt_sort_kind k, ...);
  virtual literalt mk_lit(const smt_ast *s);
  virtual smt_ast *mk_smt_int(const mp_integer &theint, bool sign);
  virtual smt_ast *mk_smt_real(const std::string &str);
  virtual smt_ast *mk_smt_bvint(const mp_integer &theint, bool sign,
                                unsigned int w);
  virtual smt_ast *mk_smt_bool(bool val);
  virtual smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s);
  virtual smt_sort *mk_struct_sort(const type2tc &type) ;
  virtual smt_sort *mk_union_sort(const type2tc &type);
  virtual smt_ast *mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s);
};
