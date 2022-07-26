/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/typecheck.h>


bool typecheckt::typecheck_main()
{
  try
  {
    typecheck();
  }

  catch(int e)
  {
    error("{}", e);
  }

  catch(const char *e)
  {
    error(e);
  }

  catch(const std::string &e)
  {
    error(e);
  }

  return error_found;
}
