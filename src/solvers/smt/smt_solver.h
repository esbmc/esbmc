#ifndef _ESBMC_PROP_SMT_SMT_SOLVER_H_
#define _ESBMC_PROP_SMT_SMT_SOLVER_H_

#include <solvers/smt/smt_conv.h>

// Transitional name for the full solver implementation base.
//
// The public conversion API keeps the smt_solver_baset name while solver backends move
// to smt_solver_baset. Once the equation conversion loop is moved out of
// goto-symex, smt_solver_baset can shrink to the minimal frontend-facing interface and
// smt_solver_baset can own the AST/sort-heavy solver implementation.

#endif
