
#ifndef SIMPLIFICATION_CHECK_H
#define SIMPLIFICATION_CHECK_H
#include "namespace.h"

#include <irep2/irep2_expr.h>

class simplification_check
{
public:
  /**
   * Check if `old_expr` and `new_expr` are equivalent where `new_expr` is the
   * result of some transformation on `old_expr`.
   *
   * @param old_expr the old expression
   * @param new_expr the new expression
   * @param ns
   */
  static void check_equivalence(
    const expr2tc &old_expr,
    const expr2tc &new_expr,
    const namespacet &ns);
};

#endif // SIMPLIFICATION_CHECK_H
