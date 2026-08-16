#if defined(_WIN32)

#else
#  include <csignal>
#  include <cstdlib>
#  include <unistd.h>
#endif

#include <util/base/filesystem.h>
#include <util/base/signal_catcher.h>

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

// Same async-signal-safety constraint as timeout_handler: this can interrupt
// the allocator, so it uses only the signal-safe cleanup paths, and _exit()
// rather than exit(), which would run atexit handlers and destructors (#6201).
void signal_catcher(int sig)
{
#if defined(_WIN32)
#else
  // kill any children by killing group
  killpg(0, sig);
  // External solvers spawned into their own process groups are not in our
  // group, so kill them explicitly.
  file_operations::kill_registered_pgroups_from_signal();

  file_operations::remove_registered_tmps_from_signal();
  _exit(sig);
#endif
}
