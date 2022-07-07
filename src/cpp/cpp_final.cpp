/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <ansi-c/c_final.h>
#include <c2goto/cprover_library.h>
#include <cpp/cpp_final.h>

bool cpp_final(contextt &context)
{
  add_cprover_library(context);

  return false;
}
