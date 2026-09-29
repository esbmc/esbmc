// #7796: delivering a signal may reset its disposition to SIG_DFL
// (C11 7.14.1.1p3), so the handler need not still be installed.
#include <csignal>
#include <cassert>

void h(int)
{
}

int main()
{
  std::signal(SIGINT, h);
  std::raise(SIGINT);
  assert(std::signal(SIGINT, h) == h);
  return 0;
}
