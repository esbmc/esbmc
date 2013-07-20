#ifndef _ESBMC_SOLVERS_SMTLIB_CONV_H_
#define _ESBMC_SOLVERS_SMTLIB_CONV_H_

#include <limits.h>
// For the sake of...
#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS
#include <stdint.h>
#include <inttypes.h>

#include <solvers/smt/smt_conv.h>
#include <solvers/smt/array_conv.h>
#include <solvers/smt/bitblast_conv.h>

#include <core/Solver.h>

typedef Minisat::Lit Lit;
typedef Minisat::lbool lbool;
typedef std::vector<literalt> bvt; // sadface.jpg

class minisat_smt_sort : public smt_sort {
  // Record all the things.
  public:
#define minisat_sort_downcast(x) static_cast<const minisat_smt_sort*>(x)

  minisat_smt_sort(smt_sort_kind i)
    : smt_sort(i), width(0), sign(false), arrdom_width(0), arrrange_width(0)
  { }

  minisat_smt_sort(smt_sort_kind i, unsigned int _width, bool _sign)
    : smt_sort(i), width(_width), sign(_sign), arrdom_width(0),
      arrrange_width(0)
  { }

  minisat_smt_sort(smt_sort_kind i, unsigned int arrwidth,
                   unsigned int rangewidth)
    : smt_sort(i), width(0), sign(false), arrdom_width(arrwidth),
      arrrange_width(rangewidth)
  { }

  virtual ~minisat_smt_sort() { }
  unsigned int width; // bv width
  bool sign;
  unsigned int arrdom_width, arrrange_width; // arr sort widths

  virtual unsigned long get_domain_width(void) const {
    return arrdom_width;
  }

  virtual unsigned long get_range_width(void) const {
    return arrrange_width;
  }

  bool is_unbounded_array(void) {
    if (id != SMT_SORT_ARRAY)
      return false;

    if (get_domain_width() > 10)
      // This is either really large, or unbounded thus leading to a machine_int
      // sized domain. Either way, not a normal one.
      return true;
    else
      return false;
  }
};

class minisat_smt_ast : public smt_ast {
public:
#define minisat_ast_downcast(x) static_cast<const minisat_smt_ast*>(x)
  minisat_smt_ast(const smt_sort *s) : smt_ast(s) { }

  // Everything is, to a greater or lesser extend, a vector of booleans, which
  // we'll represent as minisat Lit's.
  bvt bv;
};

class minisat_convt : public virtual array_convt, public virtual bitblast_convt {
public:
  typedef hash_map_cont<std::string, const smt_ast *, std::hash<std::string> > symtable_type;

  typedef enum {
    LEFT, LRIGHT, ARIGHT
  } shiftt;

  minisat_convt(bool int_encoding, const namespacet &_ns, bool is_cpp,
                const optionst &opts);
  ~minisat_convt();

  // Things definitely to be done by the solver:
  virtual resultt dec_solve();
  virtual const std::string solver_text();
  virtual tvt l_get(literalt l);
  virtual literalt new_variable();
  virtual void assert_lit(const literalt &l);
  virtual void lcnf(const bvt &bv);

  void setto(literalt a, bool val);

  // Implemented by solver for arrays:
  virtual void assign_array_symbol(const std::string &str, const smt_ast *a);

  // SMT api things that bitblast:
  virtual smt_ast* mk_func_app(const smt_sort *ressort, smt_func_kind f,
                               const smt_ast* const* args, unsigned int num);
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

  // Internal gunk

  void convert(const bvt &bv, Minisat::vec<Lit> &dest);
  void dump_bv(const bvt &bv) const;

  // What's basically the literalt api:
  // Things imported from CBMC, more or less
  literalt lnot(literalt a);
  literalt lselect(literalt a, literalt b, literalt c);
  literalt lequal(literalt a, literalt b);
  literalt limplies(literalt a, literalt b);
  literalt lxor(literalt a, literalt b);
  literalt lor(literalt a, literalt b);
  literalt land(literalt a, literalt b);
  void gate_xor(literalt a, literalt b, literalt o);
  void gate_or(literalt a, literalt b, literalt o);
  void gate_and(literalt a, literalt b, literalt o);
  void set_equal(literalt a, literalt b);

  // Members

  Minisat::Solver solver;
  const optionst &options;
  symtable_type sym_table;
};

#endif /* _ESBMC_SOLVERS_SMTLIB_CONV_H_ */
