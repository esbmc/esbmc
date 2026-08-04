#ifndef _ESBMC_PROP_SMT_SMT_RESULT_H_
#define _ESBMC_PROP_SMT_SMT_RESULT_H_

/** Result of a call to dec_solve. Either sat, unsat, or error. SMTLIB is a
 *  historic case used by the SMTLIB backend. */
enum smt_resultt
{
  P_UNSATISFIABLE,
  P_SATISFIABLE,
  P_ERROR,
  P_SMTLIB
};

#endif
