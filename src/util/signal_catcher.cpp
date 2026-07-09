#if defined(_WIN32)

#else
#  include <csignal>
#  include <cstdlib>
#endif

#include <util/filesystem.h>
#include <util/signal_catcher.h>

void install_signal_catcher()
{
#if defined(_WIN32)
#else
  // declare act to deal with action on signal set
  static struct sigaction act;

  act.sa_handler = signal_catcher;
  act.sa_flags = 0;
  sigfillset(&(act.sa_mask));

  // install signal handler
  sigaction(SIGTERM, &act, nullptr);
#endif
}

void signal_catcher(int sig)
{
#if defined(_WIN32)
#else
  // kill any children by killing group
  killpg(0, sig);
  // External solvers spawned into their own process groups are not in our
  // group, so kill them explicitly.
  file_operations::kill_registered_pgroups();

  file_operations::cleanup_registered_tmps();
  exit(sig);
#endif
}
