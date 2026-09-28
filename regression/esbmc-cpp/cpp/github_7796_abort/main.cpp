// #7796: raising SIGABRT at its default disposition ends the program.
#include <csignal>
#include <cassert>

int main()
{
  std::signal(SIGABRT, SIG_DFL);
  std::raise(SIGABRT);
  assert(0);
  return 0;
}
