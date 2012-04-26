/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include "prop.h"

void propt::set_equal(literalt a, literalt b)
{
  bvt bv;
  bv.resize(2);
  bv[0]=a;
  bv[1]=lnot(b);
  lcnf(bv);
  bv[0]=lnot(a);
  bv[1]=b;
  lcnf(bv);
}

void propt::set_assignment(literalt a, bool value)
{
  assert(false);
}

void propt::copy_assignment_from(const propt &src)
{
  assert(false);
}

bool propt::is_in_conflict(literalt l) const
{
  assert(false);
  return false;
}

bool propt::is_in_core(literalt l) const
{
  assert(false);
  return false;
}

bvt propt::new_variables(unsigned width)
{
  bvt result;
  result.resize(width);
  for(unsigned i=0; i<width; i++)
    result[i]=new_variable();
  return result;
}
