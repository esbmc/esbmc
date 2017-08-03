#ifndef _ESBMC_SOLVERS_YICES_YICES_CONV_H_
#define _ESBMC_SOLVERS_YICES_YICES_CONV_H_

#include <solvers/smt/smt_conv.h>
#include <yices.h>

class yices_smt_sort : public smt_sort
{
public:
#define yices_sort_downcast(x) static_cast<const yices_smt_sort *>(x)
  yices_smt_sort(smt_sort_kind i, type_t _s)
    : smt_sort(i), s(_s), rangesort(nullptr) { }

  yices_smt_sort(smt_sort_kind i, type_t _s, const type2tc &_tupletype)
    : smt_sort(i), s(_s), rangesort(nullptr), tupletype(_tupletype) { }

  yices_smt_sort(smt_sort_kind i, type_t _s, size_t w)
    : smt_sort(i, w), s(_s), rangesort(nullptr) { }

  yices_smt_sort(smt_sort_kind i, type_t _s, size_t w, size_t sw)
    : smt_sort(i, w, sw), s(_s), rangesort(nullptr) { }

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
  ~yices_smt_ast() override = default;

  // Provide assign semantics for arrays. While yices will swallow array
  // equalities, it appears to silently not honour them? From observation.
  void assign(smt_convt *ctx, smt_astt sym) const override;
  smt_astt project(smt_convt *ctx, unsigned int elem) const override;
  smt_astt update(smt_convt *ctx, smt_astt value,
                                unsigned int idx,
                                expr2tc idx_expr = expr2tc()) const override;
  smt_astt select(smt_convt *ctx, const expr2tc &idx) const override;
  void dump() const override { abort(); }

  term_t term;
  std::string symname;
};

class yices_convt : public smt_convt, public array_iface, public tuple_iface, public fp_convt
{
public:
  yices_convt(bool int_encoding, const namespacet &ns);
  ~yices_convt() override;

  resultt dec_solve() override;
  const std::string solver_text() override;

  void assert_ast(const smt_ast *a) override;

  smt_astt mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast * const *args,
                               unsigned int numargs) override;
  smt_sortt mk_sort(const smt_sort_kind k, ...) override;
  smt_astt mk_smt_int(const mp_integer &theint, bool sign) override;
  smt_astt mk_smt_real(const std::string &str) override;
  smt_astt mk_smt_bvint(
    const mp_integer &theint,
    bool sign,
    unsigned int w) override;
  smt_astt mk_smt_bool(bool val) override;
  smt_astt mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  smt_astt mk_array_symbol(
    const std::string &name,
    const smt_sort *s,
    smt_sortt array_subtype) override;
  smt_astt mk_extract(
    const smt_ast *a,
    unsigned int high,
    unsigned int low,
    const smt_sort *s) override;

  void push_ctx() override;
  void pop_ctx() override;

  smt_astt convert_array_of(smt_astt init_val, unsigned long domain_width) override;

  void add_array_constraints_for_solving() override;
  void push_array_ctx() override;
  void pop_array_ctx() override;

  smt_sortt mk_struct_sort(const type2tc &type) override;
  smt_astt tuple_create(const expr2tc &structdef) override;
  smt_astt tuple_fresh(smt_sortt s, std::string name = "") override;
  smt_astt tuple_array_create(const type2tc &array_type,
                              smt_astt *inputargs,
                              bool const_array,
                              smt_sortt domain) override;
  smt_astt tuple_array_of(const expr2tc &init_value,
                                              unsigned long domain_width) override;
  smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s) override;
  smt_astt mk_tuple_array_symbol(const expr2tc &expr) override;
  expr2tc tuple_get(const expr2tc &expr) override;
  void add_tuple_constraints_for_solving() override;
  void push_tuple_ctx() override;
  void pop_tuple_ctx() override;

  expr2tc get_bool(const smt_ast *a) override;
  expr2tc get_bv(const type2tc &type, smt_astt a) override;
  expr2tc get_array_elem(
    const smt_ast *array,
    uint64_t index,
    const type2tc &subtype) override;

  inline smt_astt new_ast(smt_sortt s, term_t t) {
    return new yices_smt_ast(this, s, t);
  }

  inline void clear_model() {
    if (sat_model) {
      yices_free_model(sat_model);
      sat_model = nullptr;
    }
  }

  context_t *yices_ctx;
  model_t *sat_model;
};

#endif /* _ESBMC_SOLVERS_YICES_YICES_CONV_H_ */
