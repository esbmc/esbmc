#include "simplify_expr2.h"

bool
simplify_expr2(expr2t &expr)
{
  bool res = false;
  // Base simplify - the obj being simplified has no function that simplifies
  // itself. So, fetch a vector of operands and apply simplification to anything
  // further down.
  std::vector<expr2tc> operands;
  expr.list_operands(operands);
  Forall_exprs(it, operands)
    res |= (*it).get()->simplify();
  return res;
}
