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

#endif /* SOLVERS_SMT_SMT_SORT_H_ */
