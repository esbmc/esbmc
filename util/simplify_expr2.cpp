#include "irep2.h"

expr2t *
expr2t::do_simplify(void)
{
  bool res = false;
  // Base simplify - the obj being simplified has no function that simplifies
  // itself. So, fetch a vector of operands and apply simplification to anything
  // further down.
  std::vector<expr2tc*> operands;
  list_operands(operands);
  for (std::vector<expr2tc *>::iterator it = operands.begin();
       it != operands.end(); it++) {
    expr2t *tmp = (**it).get()->do_simplify();
    if (tmp == NULL) {
      ; // No modification
    } else {
      // We've been returned a new irep to use. Assign into this objects
      // expr2tc field, which will also cause the previous value to be
      // de-referenced.
      **it = expr2tc(tmp);
    }
  }
  return false;
}
