// #7796: a signal raised in its own handler and blocked there is delivered
// once the handler returns (C11 7.14.1.1p3).
#include <csignal>
#include <cassert>

static volatile std::sig_atomic_t count = 0, x = 0;

void h(int sig)
{
  if (count == 0)
  {
    count = 1;
    std::raise(sig);
    x = 1;
  }
  else
    assert(x == 0);
}

int main()
{
  std::signal(SIGINT, h);
  std::raise(SIGINT);
  return 0;
}
