#include <util/typecheck.h>
#include <util/message.h>

bool typecheckt::typecheck_main()
{
  try
  {
    typecheck();
  }

  catch (int e)
  {
    log_error("{}", e);
    abort();
  }

  catch (const char *e)
  {
    log_error("{}", e);
    abort();
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    abort();
  }

  return false;
}
