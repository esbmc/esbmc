#ifndef _ESBMC_SOLVERS_YICES_YICES_CONV_H_
#define _ESBMC_SOLVERS_YICES_YICES_CONV_H_

#include <solvers/smt/smt_conv.h>

#include <yices.h>

class yices_smt_sort : public smt_sort
{
public:
#define yices_sort_downcast(x) static_cast<const yices_smt_sort *>(x)
  yices_smt_sort(smt_sort_kind i, type_t _s)
    : smt_sort(i), s(_s), rangesort(NULL) { }

  yices_smt_sort(smt_sort_kind i, type_t _s, const type2tc &_tupletype)
    : smt_sort(i), s(_s), rangesort(NULL), tupletype(_tupletype) { }

  yices_smt_sort(smt_sort_kind i, type_t _s, size_t w)
    : smt_sort(i, w), s(_s), rangesort(NULL) { }

  yices_smt_sort(smt_sort_kind i, type_t _s, size_t w, size_t sw)
    : smt_sort(i, w, sw), s(_s), rangesort(NULL) { }

  yices_smt_sort(smt_sort_kind i, type_t _s, size_t w, size_t dw,
                 const smt_sort *_rangesort)
    : smt_sort(i, w, dw), s(_s), rangesort(_rangesort) { }

  virtual ~yices_smt_sort() = default;

  type_t s;
  const smt_sort *rangesort;
  type2tc tupletype;
};

class yices_smt_ast : public smt_ast
{
public:
#define yices_ast_downcast(x) static_cast<const yices_smt_ast *>(x)
  yices_smt_ast(smt_convt *ctx, const smt_sort *_s, term_t _t)
    : smt_ast(ctx, _s), term(_t) {

      // Detect term errors
      if (term == NULL_TERM) {
        std::cerr << "Error creating yices term" << std::endl;
        yices_print_error(stderr);
        abort();
      }
    }
  virtual ~yices_smt_ast() { }

  // Provide assign semantics for arrays. While yices will swallow array
  // equalities, it appears to silently not honour them? From observation.
  virtual void assign(smt_convt *ctx, smt_astt sym) const;
  virtual smt_astt project(smt_convt *ctx, unsigned int elem) const;
  virtual smt_astt update(smt_convt *ctx, smt_astt value,
                                unsigned int idx,
                                expr2tc idx_expr = expr2tc()) const;
  virtual smt_astt select(smt_convt *ctx, const expr2tc &idx) const;
  virtual void dump() const { abort(); }

  term_t term;
  std::string symname;
};

class yices_convt : public smt_convt, public array_iface, public tuple_iface, public fp_convt
{
public:
  yices_convt(bool int_encoding, const namespacet &ns);
  virtual ~yices_convt();

  virtual resultt dec_solve();
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
  virtual smt_astt mk_array_symbol(const std::string &name, const smt_sort *s,
                                   smt_sortt array_subtype);
  virtual smt_astt mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s);

  virtual void push_ctx(void);
  virtual void pop_ctx(void);

  smt_astt convert_array_of(smt_astt init_val, unsigned long domain_width);

  virtual void add_array_constraints_for_solving();
  void push_array_ctx(void);
  void pop_array_ctx(void);

  virtual smt_sortt mk_struct_sort(const type2tc &type);
  virtual smt_astt tuple_create(const expr2tc &structdef);
  virtual smt_astt tuple_fresh(smt_sortt s, std::string name = "");
  virtual smt_astt tuple_array_create(const type2tc &array_type,
                              smt_astt *inputargs,
                              bool const_array,
                              smt_sortt domain);
  virtual smt_astt tuple_array_of(const expr2tc &init_value,
                                              unsigned long domain_width);
  virtual smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s);
  virtual smt_astt mk_tuple_array_symbol(const expr2tc &expr);
  virtual expr2tc tuple_get(const expr2tc &expr);
  virtual void add_tuple_constraints_for_solving();
  virtual void push_tuple_ctx();
  virtual void pop_tuple_ctx();

  virtual expr2tc get_bool(const smt_ast *a);
  virtual BigInt get_bv(const smt_ast *a);
  virtual expr2tc get_array_elem(
    const smt_ast *array,
    uint64_t index,
    const type2tc &subtype);

  inline smt_astt new_ast(smt_sortt s, term_t t) {
    return new yices_smt_ast(this, s, t);
  }

  inline void clear_model(void) {
    if (sat_model) {
      yices_free_model(sat_model);
      sat_model = NULL;
    }
  }

  context_t *yices_ctx;
  model_t *sat_model;
};

#endif /* _ESBMC_SOLVERS_YICES_YICES_CONV_H_ */
