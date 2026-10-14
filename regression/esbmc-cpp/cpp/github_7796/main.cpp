// #7796: <csignal> declares std::signal, std::raise and std::sig_atomic_t and
// the SIG* macros, and can follow <signal.h>.
#include <signal.h>
#include <csignal>
#include <cassert>
#include <cerrno>

static std::sig_atomic_t hits = 0;

void h(int sig)
{
  hits += sig;
}

int main()
{
  if (std::signal(SIGTERM, SIG_IGN) == SIG_ERR)
    return 1;
  assert(std::signal(SIGTERM, h) == SIG_IGN);
  void (*first)(int) = std::signal(SIGINT, h);
  assert(first == SIG_DFL || first == SIG_IGN);
  assert(std::raise(SIGINT) == 0);
  assert(hits == SIGINT);
  void (*after)(int) = std::signal(SIGINT, h);
  assert(after == h || after == SIG_DFL);

  errno = 0;
  assert(std::signal(0, h) == SIG_ERR && errno != 0);
  assert(std::signal(NSIG, h) == SIG_ERR);
  assert(std::signal(SIGTERM, SIG_ERR) == SIG_ERR);
#ifdef SIGKILL
  assert(std::signal(SIGKILL, h) == SIG_ERR);
#endif
  assert(std::raise(0) == 0 && hits == SIGINT);
  errno = 0;
  assert(std::raise(-1) != 0 && errno != 0);
  assert(std::raise(NSIG) != 0);

  std::signal(SIGTERM, SIG_IGN);
  assert(std::raise(SIGTERM) == 0 && hits == SIGINT);
  return 0;
}
