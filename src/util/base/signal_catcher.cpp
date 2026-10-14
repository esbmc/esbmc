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

void signal_safe_write(int fd, const char *msg, size_t len)
{
#if defined(_WIN32)
  (void)fd;
  (void)msg;
  (void)len;
#else
  while (len)
  {
    ssize_t written = write(fd, msg, len);
    if (written <= 0)
      return;
    msg += written;
    len -= (size_t)written;
  }
#endif
}

#if !defined(_WIN32)
namespace
{
/// Set by install_fatal_signal_reporter; read from the handler, where anything
/// that allocates or locks is off-limits.
const char *fatal_signal_advice = nullptr;

void fatal_signal_reporter(int sig)
{
  static const char head[] = "\nESBMC caught ";
  static const char segv[] = "SIGSEGV";
  static const char bus[] = "SIGBUS";
  static const char body[] =
    ": this is an internal error, not a verification result.\n";
  static const char report[] =
    "Please report it at https://github.com/esbmc/esbmc/issues\n";

  // Assembled in one buffer and emitted in a single write: another thread is
  // very likely still logging, and separate writes interleave with its output
  // mid-sentence.
  char msg[512];
  size_t len = 0;
  auto append = [&msg, &len](const char *text, size_t n) {
    if (n > sizeof(msg) - len)
      n = sizeof(msg) - len;
    for (size_t i = 0; i < n; i++)
      msg[len + i] = text[i];
    len += n;
  };

  append(head, sizeof(head) - 1);
  if (sig == SIGBUS)
    append(bus, sizeof(bus) - 1);
  else
    append(segv, sizeof(segv) - 1);
  append(body, sizeof(body) - 1);
  if (fatal_signal_advice)
  {
    size_t advice_len = 0;
    while (fatal_signal_advice[advice_len])
      advice_len++;
    append(fatal_signal_advice, advice_len);
  }
  append(report, sizeof(report) - 1);

  signal_safe_write(STDERR_FILENO, msg, len);

  // Die from the original signal so the exit status still says what happened.
  ::signal(sig, SIG_DFL);
  ::raise(sig);
}

/// Give this thread an alternate signal stack, unless it already has one: a
/// sanitizer installs its own, and stealing it breaks its reporting.
bool ensure_alt_stack()
{
  static thread_local int state = 0; // 0 untried, 1 available, -1 unavailable
  if (state)
    return state > 0;

  stack_t current;
  if (sigaltstack(nullptr, &current) == 0 && !(current.ss_flags & SS_DISABLE))
  {
    state = 1;
    return true;
  }

  stack_t ss;
  ss.ss_size = 64 * 1024;
  ss.ss_sp = malloc(ss.ss_size);
  ss.ss_flags = 0;
  state = ss.ss_sp && sigaltstack(&ss, nullptr) == 0 ? 1 : -1;
  return state > 0;
}

bool disposition_is_default(int sig)
{
  struct sigaction old = {};
  // sa_handler is only the live union member when SA_SIGINFO is clear.
  return sigaction(sig, nullptr, &old) == 0 && !(old.sa_flags & SA_SIGINFO) &&
         old.sa_handler == SIG_DFL;
}
} // namespace
#endif

bool install_altstack_handler(int sig, void (*handler)(int), bool keep_existing)
{
#if defined(_WIN32)
  (void)sig;
  (void)handler;
  (void)keep_existing;
  return false;
#else
  if (keep_existing && !disposition_is_default(sig))
    return false;

  struct sigaction act = {};
  act.sa_handler = handler;
  act.sa_flags = ensure_alt_stack() ? SA_ONSTACK : 0;
  sigemptyset(&(act.sa_mask));

  return sigaction(sig, &act, nullptr) == 0;
#endif
}

void install_fatal_signal_reporter(const char *extra_advice)
{
#if defined(_WIN32)
  (void)extra_advice;
#else
  fatal_signal_advice = extra_advice;
  // The alternate stack is per-thread, so this has to be called on the thread
  // that runs the work -- deep recursion over expression structure is one of
  // the ways ESBMC faults (#6617), and that happens on the worker thread the
  // driver's main() runs on, not on the process's main thread. Threads spawned
  // later (--parallel-solving, clang's) get no alternate stack, so a
  // stack-exhaustion fault on one of those is still silent.
  //
  // keep_existing: a sanitizer's own SIGSEGV handler prints far more than this
  // one does, so never displace it.
  install_altstack_handler(SIGSEGV, fatal_signal_reporter, true);
  install_altstack_handler(SIGBUS, fatal_signal_reporter, true);
#endif
}
