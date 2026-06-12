#ifndef _ESBMC_PROP_SMT_SMT_API_H_
#define _ESBMC_PROP_SMT_SMT_API_H_

/** @file smt_api.h
 *  Minimal public SMT API for layers that need solver interaction without
 *  depending on solver internals.
 */

class smt_convt;

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
