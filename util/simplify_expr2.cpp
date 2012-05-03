#include "irep2.h"

bool
expr2t::do_simplify(void)
{
  bool res = false;
  // Base simplify - the obj being simplified has no function that simplifies
  // itself. So, fetch a vector of operands and apply simplification to anything
  // further down.
  std::vector<expr2tc> operands;
  list_operands(operands);
  Forall_exprs(it, operands)
    res |= (*it).get()->simplify();
  return res;
}
