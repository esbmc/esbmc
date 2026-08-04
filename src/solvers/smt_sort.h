#ifndef SOLVERS_SMT_SMT_SORT_H_
#define SOLVERS_SMT_SMT_SORT_H_

#include <camada/camada.h>

/** An SMT sort: camada's own handle, not a wrapper around it.
 *
 *  Camada already records the kind, the widths, an array's index and element
 *  sorts and a tuple's element sorts, and answers isArithSort() and friends, so
 *  a second object holding copies of all that would cost an allocation per sort
 *  and a chance to disagree. Use the handle's own methods directly.
 *
 *  The handle is nullable (`if (!sort)`) and validated against the owning
 *  solver's generation, so one outliving its solver aborts rather than reading
 *  freed memory.
 */
using smt_sortt = camada::SMTSortRef;

/** An SMT function application: camada's own handle, for the same reasons.
 *
 *  This was an ESBMC-side class with six virtuals, but every implementation
 *  either forwarded straight back into the solver context or dispatched on the
 *  operand's sort, so none of them needed a vtable or an allocation. They are
 *  now smt_solver_baset::ast_eq / ast_assign / ast_update / ast_select /
 *  ast_project. A camada expression carries its own sort, so nothing here has
 *  to store one.
 */
using smt_astt = camada::SMTExprRef;

#endif /* SOLVERS_SMT_SMT_SORT_H_ */
