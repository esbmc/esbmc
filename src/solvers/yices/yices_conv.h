#ifndef _ESBMC_SOLVERS_YICES_YICES_CONV_H_
#define _ESBMC_SOLVERS_YICES_YICES_CONV_H_

#include <solvers/smt/smt_conv.h>
#include <yices.h>

class yices_smt_sort : public smt_sort
{
public:
#define yices_sort_downcast(x) static_cast<const yices_smt_sort *>(x)
  yices_smt_sort(smt_sort_kind i, type_t _t)
    : smt_sort(i), type(_t), tuple_type(), arr_range(NULL) { }
  yices_smt_sort(smt_sort_kind i, type_t _t, unsigned int w)
    : smt_sort(i, w), type(_t), tuple_type(), arr_range(NULL) { }

  yices_smt_sort(smt_sort_kind i, type_t _t, unsigned long w,
                 unsigned long d, const yices_smt_sort *rangetype)
    : smt_sort(i, w, d), type(_t), tuple_type(), arr_range(rangetype) { }

  // Constructor for structs. Bitwidth is set to 1 as an estople
  // that... it's a valid domain sort.
  yices_smt_sort(smt_sort_kind i, type_t _t, const type2tc &s)
    : smt_sort(i, 1), type(_t), tuple_type(s), arr_range(NULL) { }

  virtual ~yices_smt_sort() { }

  type_t type;
  type2tc tuple_type; // Only valid for tuples
  const yices_smt_sort *arr_range;
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

class yices_convt : public smt_convt, public array_iface, public tuple_iface
{
public:
  yices_convt(bool int_encoding, const namespacet &ns);
  virtual ~yices_convt();

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
  virtual smt_astt mk_smt_bvfloat(const ieee_floatt &thereal,
                                  unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_nan(unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm);
  virtual smt_astt mk_smt_typecast_from_bvfloat(const typecast2t &cast);
  virtual smt_astt mk_smt_typecast_to_bvfloat(const typecast2t &cast);
  virtual smt_astt mk_smt_nearbyint_from_float(const nearbyint2t &expr);
  virtual smt_astt mk_smt_bvfloat_arith_ops(const expr2tc &expr);
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

  virtual expr2tc tuple_get_rec(term_t term, const type2tc &type);

  expr2tc get_bool(smt_astt a);
  expr2tc get_bv(const type2tc &t, smt_astt a);
  expr2tc get_array_elem(smt_astt array, uint64_t index,
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
