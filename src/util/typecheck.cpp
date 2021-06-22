/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/typecheck.h>
#include <util/message/format.h>

bool typecheckt::typecheck_main()
{
  try
  {
    typecheck();
  }

  catch(int e)
  {
    error(fmt::format("{}", e));
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
