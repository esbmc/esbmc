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
};

#endif /* _ESBMC_SOLVERS_SMT_BITBLAST_CONV_H_ */
