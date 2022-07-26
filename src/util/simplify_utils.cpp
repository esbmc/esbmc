#include <algorithm>
#include <util/simplify_utils.h>

bool sort_operands(exprt::operandst &operands)
{
  bool do_sort = false;

  forall_expr(it, operands)
  {
    exprt::operandst::const_iterator next_it = it;
    next_it++;

    if(next_it != operands.end() && *next_it < *it)
    {
      do_sort = true;
      break;
    }
  }

  if(!do_sort)
    return true;

  std::sort(operands.begin(), operands.end());

  return false;
}
