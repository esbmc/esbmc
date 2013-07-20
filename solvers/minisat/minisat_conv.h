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

class minisat_convt : public virtual array_convt, public virtual bitblast_convt {
public:
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
};

#endif /* _ESBMC_SOLVERS_SMTLIB_CONV_H_ */
