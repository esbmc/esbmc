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

#if !defined(_WIN32)
namespace
{
void write_all(int fd, const char *msg, size_t len)
{
  while (len)
  {
    ssize_t written = write(fd, msg, len);
    if (written <= 0)
      return;
    msg += written;
    len -= (size_t)written;
  }
}

void fatal_signal_reporter(int sig)
{
  static const char segv[] =
    "\nESBMC caught SIGSEGV: this is an internal error, not a verification "
    "result.\nRe-run with --segfault-handler for a backtrace and report it at "
    "https://github.com/esbmc/esbmc/issues\n";
  static const char bus[] =
    "\nESBMC caught SIGBUS: this is an internal error, not a verification "
    "result.\nRe-run with --segfault-handler for a backtrace and report it at "
    "https://github.com/esbmc/esbmc/issues\n";

  const char *msg = sig == SIGBUS ? bus : segv;
  write_all(
    STDERR_FILENO, msg, (sig == SIGBUS ? sizeof(bus) : sizeof(segv)) - 1);

  // Die from the original signal so the exit status still says what happened.
  ::signal(sig, SIG_DFL);
  ::raise(sig);
}
} // namespace
#endif

void install_fatal_signal_reporter()
{
#if defined(_WIN32)
#else
  // A handler cannot run on the stack whose exhaustion raised the signal, and
  // deep recursion over expression structure is one of the ways ESBMC faults
  // (#6617). The alternate stack is per-thread, so this has to be called on
  // the thread that does the work, not on main's.
  static thread_local char alt_stack[64 * 1024];
  stack_t ss;
  ss.ss_sp = alt_stack;
  ss.ss_size = sizeof(alt_stack);
  ss.ss_flags = 0;
  sigaltstack(&ss, nullptr);

  struct sigaction act;
  act.sa_handler = fatal_signal_reporter;
  act.sa_flags = SA_ONSTACK;
  sigemptyset(&(act.sa_mask));

  sigaction(SIGSEGV, &act, nullptr);
  sigaction(SIGBUS, &act, nullptr);
#endif
}
