// #7796: std::raise runs the handler std::signal installed.
#include <csignal>
#include <cassert>

static std::sig_atomic_t hits = 0;

void h(int sig)
{
  hits += sig;
}

int main()
{
  std::signal(SIGINT, h);
  std::raise(SIGINT);
  assert(hits == 0);
  return 0;
}
