#ifndef _ESBMC_SOLVERS_SMTLIB_CONV_H_
#define _ESBMC_SOLVERS_SMTLIB_CONV_H_

#include <limits.h>
// For the sake of...
#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS
#include <stdint.h>
#include <inttypes.h>

#include <solvers/smt/smt_conv.h>

#include <core/Solver.h>

typedef Minisat::Lit Lit;
typedef Minisat::lbool lbool;
typedef std::vector<literalt> bvt; // sadface.jpg

class minisat_smt_sort : public smt_sort {
  // Record all the things.
  public:
#define minisat_sort_downcast(x) static_cast<const minisat_smt_sort*>(x)

  minisat_smt_sort(smt_sort_kind i)
    : smt_sort(i), width(0), arrdom_width(0), arrrange_width(0)
  { }

  minisat_smt_sort(smt_sort_kind i, unsigned int _width)
    : smt_sort(i), width(_width), arrdom_width(0), arrrange_width(0)
  { }

  minisat_smt_sort(smt_sort_kind i, unsigned int arrwidth,
                   unsigned int rangewidth)
    : smt_sort(i), width(0), arrdom_width(arrwidth), arrrange_width(rangewidth)
  { }

  virtual ~minisat_smt_sort() { }
  unsigned int width; // bv width
  unsigned int arrdom_width, arrrange_width; // arr sort widths

  virtual unsigned long get_domain_width(void) const {
    return arrdom_width;
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

class minisat_convt : public smt_convt {
public:
  typedef hash_map_cont<std::string, const minisat_smt_ast *, std::hash<std::string> > symtable_type;

  minisat_convt(bool int_encoding, const namespacet &_ns, bool is_cpp,
                const optionst &opts);
  ~minisat_convt();

  virtual resultt dec_solve();
  virtual expr2tc get(const expr2tc &expr);
  virtual const std::string solver_text();
  virtual tvt l_get(literalt l);
  virtual void assert_lit(const literalt &l);
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

  virtual literalt new_variable();

  minisat_smt_ast *mk_ast_equality(const minisat_smt_ast *a,
                                   const minisat_smt_ast *b);

  // Things imported from CBMC, more or less
  void convert(const bvt &bv, Minisat::vec<Lit> &dest);
  literalt lnot(literalt a);
  literalt lequal(literalt a, literalt b);
  literalt lxor(literalt a, literalt b);
  literalt lor(literalt a, literalt b);
  void gate_xor(literalt a, literalt b, literalt o);
  void gate_or(literalt a, literalt b, literalt o);
  void set_equal(literalt a, literalt b);
  virtual void lcnf(const bvt &bv);

  Minisat::Solver solver;
  const optionst &options;
  symtable_type sym_table;
};

#endif /* _ESBMC_SOLVERS_SMTLIB_CONV_H_ */
