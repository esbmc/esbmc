// #7796: a signal's first disposition may be inherited as SIG_IGN, so the
// first std::signal need not return SIG_DFL (C11 7.14.1.1p6).
#include <csignal>
#include <cassert>

void h(int)
{
}

int main()
{
  assert(std::signal(SIGINT, h) == SIG_DFL);
  return 0;
}
