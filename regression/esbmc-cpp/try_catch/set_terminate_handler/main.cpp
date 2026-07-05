#include <exception>
#include <cstdlib>

// std::set_terminate installs a handler that std::terminate must invoke. Here
// the handler exits cleanly, so the program terminates without the default
// "terminate called" failure. (Previously set_terminate was stubbed and the
// handler was silently ignored -> spurious failure.)
void my_terminate()
{
  std::exit(0);
}

int main()
{
  std::set_terminate(my_terminate);
  std::terminate();
  return 0;
}
