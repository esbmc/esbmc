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

class bitblast_convt : public virtual smt_convt
{
public:
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

  virtual smt_ast* mk_func_app(const smt_sort *ressort, smt_func_kind f,
                               const smt_ast* const* args, unsigned int num);

  // Boolean operations we require.
  virtual literalt lnot(literalt a) = 0;
  virtual literalt lselect(literalt a, literalt b, literalt c) = 0;
  virtual literalt lequal(literalt a, literalt b) = 0;
  virtual literalt limplies(literalt a, literalt b) = 0;
  virtual literalt lxor(literalt a, literalt b) = 0;
  virtual literalt lor(literalt a, literalt b) = 0;
  virtual literalt land(literalt a, literalt b) = 0;
  virtual void gate_xor(literalt a, literalt b, literalt o) = 0;
  virtual void gate_or(literalt a, literalt b, literalt o) = 0;
  virtual void gate_and(literalt a, literalt b, literalt o) = 0;
  virtual void set_equal(literalt a, literalt b) = 0;

  // Bitblasting utilities, mostly from CBMC.
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
};

#endif /* _ESBMC_SOLVERS_SMT_BITBLAST_CONV_H_ */
