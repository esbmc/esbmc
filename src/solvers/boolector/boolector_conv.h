#include <stdio.h>

#include <irep2.h>
#include <namespace.h>

#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_tuple_flat.h>

extern "C" {
#include <boolector.h>
}

// There's no boolector sort structure, therefore just use smt_sort.

class btor_smt_ast : public smt_ast
{
public:
#define btor_ast_downcast(x) static_cast<const btor_smt_ast *>(x)
  btor_smt_ast(smt_convt *ctx, const smt_sort *_s, BoolectorNode *_e)
    : smt_ast(ctx, _s), e(_e) { }
  virtual ~btor_smt_ast() { }
  virtual void dump() const { abort(); }

  BoolectorNode *e;
};

class boolector_convt : public smt_convt, public array_iface
{
public:
  typedef hash_map_cont<std::string, smt_ast *, std::hash<std::string> >
    symtable_type;

  boolector_convt(bool int_encoding, const namespacet &ns,
                  const optionst &options);
  virtual ~boolector_convt();

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
  virtual smt_ast *mk_smt_bvfloat(const ieee_floatt &thereal,
                                  unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_nan(unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm);
  virtual smt_astt mk_smt_typecast_from_bvfloat(const typecast2t &cast);
  virtual smt_astt mk_smt_typecast_to_bvfloat(const typecast2t &cast);
  virtual smt_astt mk_smt_nearbyint_from_float(const nearbyint2t &expr);
  virtual smt_astt mk_smt_bvfloat_arith_ops(const expr2tc &expr);
  virtual smt_ast *mk_smt_bool(bool val);
  virtual smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s);
  virtual smt_ast *mk_array_symbol(const std::string &name, const smt_sort *s,
                                   smt_sortt array_subtype);
  virtual smt_sort *mk_struct_sort(const type2tc &type);
  virtual smt_ast *mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s);

  const smt_ast *convert_array_of(smt_astt init_val,
                                  unsigned long domain_width);

  virtual void add_array_constraints_for_solving();
  void push_array_ctx(void);
  void pop_array_ctx(void);

  expr2tc get_bool(const smt_ast *a);
  expr2tc get_bv(const type2tc &t, const smt_ast *a);
  expr2tc get_array_elem(const smt_ast *array, uint64_t index,
                         const type2tc &subtype);

  virtual const smt_ast *overflow_arith(const expr2tc &expr);

  inline btor_smt_ast *new_ast(const smt_sort *_s, BoolectorNode *_e) {
    return new btor_smt_ast(this, _s, _e);
  }

  typedef BoolectorNode *(*shift_func_ptr)
                         (Btor *, BoolectorNode *, BoolectorNode *);
  smt_ast *fix_up_shift(shift_func_ptr fptr, const btor_smt_ast *op0,
      const btor_smt_ast *op1, smt_sortt res_sort);

  // Members

  Btor *btor;
  symtable_type symtable;
  FILE *debugfile;
};
