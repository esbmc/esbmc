#include <cstdio>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_tuple_flat.h>
#include <util/irep2.h>
#include <util/namespace.h>

extern "C" {
#include <boolector.h>
}

class boolector_smt_sort : public smt_sort
{
public:
#define boolector_sort_downcast(x) static_cast<const boolector_smt_sort *>(x)
  boolector_smt_sort(smt_sort_kind i, BoolectorSort _s)
    : smt_sort(i), s(_s), rangesort(nullptr) { }

  boolector_smt_sort(smt_sort_kind i, BoolectorSort _s, size_t w)
    : smt_sort(i, w), s(_s), rangesort(nullptr) { }

  boolector_smt_sort(smt_sort_kind i, BoolectorSort _s, size_t w, size_t sw)
    : smt_sort(i, w, sw), s(_s), rangesort(nullptr) { }

  boolector_smt_sort(smt_sort_kind i, BoolectorSort  _s, size_t w, size_t dw,
                     const smt_sort *_rangesort)
    : smt_sort(i, w, dw), s(_s), rangesort(_rangesort) { }

  virtual ~boolector_smt_sort() = default;

  BoolectorSort s;
  const smt_sort *rangesort;
};


class btor_smt_ast : public smt_ast
{
public:
#define btor_ast_downcast(x) static_cast<const btor_smt_ast *>(x)
  btor_smt_ast(smt_convt *ctx, const smt_sort *_s, BoolectorNode *_e)
    : smt_ast(ctx, _s), e(_e) { }

  const smt_ast *select(smt_convt *ctx, const expr2tc &idx) const override;

  ~btor_smt_ast() override = default;
  void dump() const override;

  BoolectorNode *e;
};

class boolector_convt : public smt_convt, public array_iface, public fp_convt
{
public:
  typedef hash_map_cont<std::string, smt_ast *, std::hash<std::string> >
    symtable_type;

  boolector_convt(bool int_encoding, const namespacet &ns,
                  const optionst &options);
  ~boolector_convt() override;

  resultt dec_solve() override;
  const std::string solver_text() override;

  void assert_ast(const smt_ast *a) override;

  smt_ast *mk_func_app(const smt_sort *s, smt_func_kind k,
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
  smt_ast *mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s) override;

  const smt_ast *convert_array_of(smt_astt init_val,
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

  const smt_ast *overflow_arith(const expr2tc &expr) override;

  inline btor_smt_ast *new_ast(const smt_sort *_s, BoolectorNode *_e) {
    return new btor_smt_ast(this, _s, _e);
  }

  typedef BoolectorNode *(*shift_func_ptr)
                         (Btor *, BoolectorNode *, BoolectorNode *);
  smt_ast *fix_up_shift(shift_func_ptr fptr, const btor_smt_ast *op0,
      const btor_smt_ast *op1, smt_sortt res_sort);

  void dump_smt() override;

  // Members

  Btor *btor;
  symtable_type symtable;
  FILE *debugfile;
};
