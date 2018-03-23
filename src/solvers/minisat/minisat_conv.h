#ifndef _ESBMC_SOLVERS_SMTLIB_CONV_H_
#define _ESBMC_SOLVERS_SMTLIB_CONV_H_

// For the sake of...
#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

#include <limits.h>
#include <stdint.h>
#include <inttypes.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_tuple_flat.h>
#include <solvers/smt/array_conv.h>
#include <solvers/sat/bitblast_conv.h>
#include <solvers/sat/cnf_conv.h>
#include <core/Solver.h>

typedef Minisat::Lit Lit;
typedef Minisat::lbool lbool;
typedef std::vector<literalt> bvt;

class minisat_convt : public cnf_iface, public cnf_convt, public bitblast_convt
{
public:
  typedef enum { LEFT, LRIGHT, ARIGHT } shiftt;

  minisat_convt(bool int_encoding, const namespacet &_ns, const optionst &opts);
  ~minisat_convt();

  // Things definitely to be done by the solver:
  virtual resultt dec_solve();
  virtual const std::string solver_text();
  virtual tvt l_get(const literalt &a);
  virtual literalt new_variable();
  virtual void assert_lit(const literalt &l);
  virtual void lcnf(const bvt &bv);

  virtual void setto(literalt a, bool val);

  // Internal gunk

  void convert(const bvt &bv, Minisat::vec<Lit> &dest);
  void dump_bv(const bvt &bv) const;

  // Members

  Minisat::Solver solver;
  const optionst &options;
  bool false_asserted;
};

#endif /* _ESBMC_SOLVERS_SMTLIB_CONV_H_ */
