#ifndef _ESBMC_SOLVERS_SMT_BITBLAST_CONV_H_
#define _ESBMC_SOLVERS_SMT_BITBLAST_CONV_H_

#include "smt_conv.h"

class bitblast_smt_sort : public smt_sort {
  // Record all the things.
  public:
#define bitblast_sort_downcast(x) static_cast<const bitblast_smt_sort*>(x)

  bitblast_smt_sort(smt_sort_kind i)
    : smt_sort(i), width(0), sign(false), arrdom_width(0), arrrange_width(0)
  { }

  bitblast_smt_sort(smt_sort_kind i, unsigned int _width, bool _sign)
    : smt_sort(i), width(_width), sign(_sign), arrdom_width(0),
      arrrange_width(0)
  { }

  bitblast_smt_sort(smt_sort_kind i, unsigned int arrwidth,
                   unsigned int rangewidth)
    : smt_sort(i), width(0), sign(false), arrdom_width(arrwidth),
      arrrange_width(rangewidth)
  { }

  virtual ~bitblast_smt_sort() { }
  unsigned int width; // bv width
  bool sign;
  unsigned int arrdom_width, arrrange_width; // arr sort widths

  virtual unsigned long get_domain_width(void) const {
    return arrdom_width;
  }

  virtual unsigned long get_range_width(void) const {
    return arrrange_width;
  }
};

class bitblast_smt_ast : public smt_ast {
public:
#define bitblast_ast_downcast(x) static_cast<const bitblast_smt_ast*>(x)
  bitblast_smt_ast(const smt_sort *s) : smt_ast(s) { }

  // Everything is, to a greater or lesser extend, a vector of booleans
  bvt bv;
};

template <class subclass>
class bitblast_convt : public virtual subclass
{
public:
  typedef hash_map_cont<std::string, smt_ast *, std::hash<std::string> >
    symtable_type;

  typedef enum {
    LEFT, LRIGHT, ARIGHT
  } shiftt;

  bitblast_convt(bool enable_cache, bool int_encoding, const namespacet &_ns,
                 bool is_cpp, bool tuple_support, bool bools_in_arrs,
                 bool can_init_inf_arrs);
  ~bitblast_convt();

  // The plan: have a mk_func_app method available, that's called by the
  // subclass when appropriate, and if there's an operation on bitvectors
  // in there, we convert it to an operation on literals, implemented using
  // the abstract api below.
  //
  // This means that the subclass relinquishes all control over both ASTs
  // and sorts: this class will manage all of that. Only operations on literals
  // will reach the subclass (via the aforementioned api), and crucially that's
  // _all_ operations. Operations on booleans from a higher level should all
  // pass through this class before becoming a literal operation.
  //
  // (This alas is not the truth yet, but it's an aim).
  //
  // The remanining flexibility options available to the solver are then only
  // in the domain of logical operations on literals, although all kinds of
  // other API things can be fudged, such as tuples and arrays.

  // Boolean operations we require.
//  virtual literalt lnot(literalt a) = 0;
//  virtual literalt lselect(literalt a, literalt b, literalt c) = 0;
//  virtual literalt lequal(literalt a, literalt b) = 0;
//  virtual literalt limplies(literalt a, literalt b) = 0;
//  virtual literalt lxor(literalt a, literalt b) = 0;

  // smt_convt apis we fufil

  virtual smt_ast* mk_func_app(const smt_sort *ressort, smt_func_kind f,
                               const smt_ast* const* args, unsigned int num);
  virtual expr2tc get(const expr2tc &expr);
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
  virtual const smt_ast *lit_to_ast(const literalt &l);

  // Some gunk
  expr2tc get_bool(const smt_ast *a);
  expr2tc get_bv(const type2tc &t, const smt_ast *a);

  // Bitblasting utilities, mostly from CBMC.
  bitblast_smt_ast *mk_ast_equality(const smt_ast *a, const smt_ast *b,
                                    const smt_sort *ressort);
  virtual literalt land(const bvt &bv);
  virtual literalt lor(const bvt &bv);
  void eliminate_duplicates(const bvt &bv, bvt &dest);
  void bvand(const bvt &bv0, const bvt &bv1, bvt &output);
  void bvor(const bvt &bv0, const bvt &bv1, bvt &output);
  void bvxor(const bvt &bv0, const bvt &bv1, bvt &output);
  void bvnot(const bvt &bv0, bvt &output);
  void full_adder(const bvt &op0, const bvt &op1, bvt &output,
                  literalt carry_in, literalt &carry_out);
  literalt carry(literalt a, literalt b, literalt c);
  literalt carry_out(const bvt &a, const bvt &b, literalt c);
  literalt equal(const bvt &op0, const bvt &op1);
  literalt lt_or_le(bool or_equal, const bvt &bv0, const bvt &bv1,
                    bool is_signed);
  void invert(bvt &bv);
  void barrel_shift(const bvt &op, const shiftt s, const bvt &dist, bvt &out);
  void shift(const bvt &inp, const shiftt &s, unsigned long d, bvt &out);
  literalt unsigned_less_than(const bvt &arg0, const bvt &arg1);
  void unsigned_multiplier(const bvt &op0, const bvt &bv1, bvt &output);
  void signed_multiplier(const bvt &op0, const bvt &bv1, bvt &output);
  void cond_negate(const bvt &vals, bvt &out, literalt cond);
  void negate(const bvt &inp, bvt &oup);
  void incrementer(const bvt &inp, const literalt &carryin, literalt carryout,
                   bvt &oup);
  void signed_divider(const bvt &op0, const bvt &op1, bvt &res, bvt &rem);
  void unsigned_divider(const bvt &op0, const bvt &op1, bvt &res, bvt &rem);
  void unsigned_multiplier_no_overflow(const bvt &op0, const bvt &op1, bvt &r);
  void adder_no_overflow(const bvt &op0, const bvt &op1, bvt &res,
                         bool subtract, bool is_signed);
  void adder_no_overflow(const bvt &op0, const bvt &op1, bvt &res);
  bool is_constant(const bvt &bv);

  // Members

  // This is placed here because (IMO) this class is connection between a bunch
  // of literals and actual things that we give names to. So it's the logical
  // place for these things to come together.
  symtable_type sym_table;
};

// And because this is a template...
#include "bitblast_conv.cpp"

#endif /* _ESBMC_SOLVERS_SMT_BITBLAST_CONV_H_ */
