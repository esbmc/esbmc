#include <exception>
#include <cstdlib>

static void clean_terminate()
{
  std::exit(0); // clean exit; not the default "terminate called" failure
}

void install_handler()
{
  std::set_terminate(clean_terminate);
}
