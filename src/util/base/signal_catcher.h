#include <cstddef>

void install_signal_catcher();
void signal_catcher(int sig);

/// Write `len` bytes without allocating or touching stdio, so it is safe from
/// a signal handler. Short writes are retried; a failure is dropped.
void signal_safe_write(int fd, const char *msg, size_t len);

/// Install `handler` for `sig` to run on an alternate signal stack, which is
/// what lets a stack-exhaustion fault run a handler at all. With
/// `keep_existing`, a disposition somebody else already set -- a sanitizer's,
/// typically -- is left alone. Returns whether the handler was installed.
bool install_altstack_handler(
  int sig,
  void (*handler)(int),
  bool keep_existing);

/// Report a fatal memory-fault signal on stderr before dying from it. Without
/// this, ESBMC's only trace of a SIGSEGV is the shell's exit status, which is
/// indistinguishable from a killed process. `extra_advice`, if given, is one
/// line naming what the *calling driver* offers for a fuller report; it must
/// outlive the process.
void install_fatal_signal_reporter(const char *extra_advice);
