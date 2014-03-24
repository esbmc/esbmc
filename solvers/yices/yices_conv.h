#ifndef _ESBMC_SOLVERS_YICES_YICES_CONV_H_
#define _ESBMC_SOLVERS_YICES_YICES_CONV_H_

#include <solvers/smt/smt_conv.h>

#include <yices.h>

class yices_smt_sort : public smt_sort
{
public:
#define yices_sort_downcast(x) static_cast<const yices_smt_sort *>(x)
  yices_smt_sort(smt_sort_kind i, type_t _t) : smt_sort(i), type(_t) { }
  yices_smt_sort(smt_sort_kind i, type_t _t, unsigned int w)
    : smt_sort(i, w), type(_t) { }
  yices_smt_sort(smt_sort_kind i, type_t _t, unsigned long w,
                 unsigned long d)
    : smt_sort(i, w, d), type(_t) { }
  virtual ~yices_smt_sort() { }

  type_t type;
};

class yices_smt_ast : public smt_ast
{
public:
#define yices_ast_downcast(x) static_cast<const yices_smt_ast *>(x)
  yices_smt_ast(smt_convt *ctx, const smt_sort *_s, term_t _t)
    : smt_ast(ctx, _s), term(_t) { }
  virtual ~yices_smt_ast() { }

  term_t term;
};

class yices_convt : public smt_convt, public array_iface
{
public:
  yices_convt(bool int_encoding, const namespacet &ns, bool is_cpp);

  virtual resultt dec_solve();
  virtual tvt l_get(const smt_ast *l);
  virtual const std::string solver_text();

  virtual void assert_ast(const smt_ast *a);

  virtual smt_astt mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast * const *args,
                               unsigned int numargs);
  virtual smt_sortt mk_sort(const smt_sort_kind k, ...);
  virtual smt_astt mk_smt_int(const mp_integer &theint, bool sign);
  virtual smt_astt mk_smt_real(const std::string &str);
  virtual smt_astt mk_smt_bvint(const mp_integer &theint, bool sign,
                                unsigned int w);
  virtual smt_astt mk_smt_bool(bool val);
  virtual smt_astt mk_smt_symbol(const std::string &name, const smt_sort *s);
  virtual smt_astt mk_array_symbol(const std::string &name, const smt_sort *s);
  virtual smt_sortt mk_struct_sort(const type2tc &type);
  virtual smt_sortt mk_union_sort(const type2tc &type);
  virtual smt_astt mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s);

  smt_astt convert_array_of(const expr2tc &init_val,
                                  unsigned long domain_width);

  virtual void add_array_constraints_for_solving();

  expr2tc get_bool(smt_astt a);
  expr2tc get_bv(const type2tc &t, smt_astt a);
  expr2tc get_array_elem(smt_astt array, uint64_t index,
                         const type2tc &subtype);

};

#endif /* _ESBMC_SOLVERS_YICES_YICES_CONV_H_ */
