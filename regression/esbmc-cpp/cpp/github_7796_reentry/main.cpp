// #7796: a signal raised in its own handler is not delivered nested: the
// disposition was reset to SIG_DFL, or the signal waits for the handler to
// return (C11 7.14.1.1p3).
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
    assert(x == 1);
}

int main()
{
  std::signal(SIGINT, h);
  std::raise(SIGINT);
  return 0;
}
