// A fatal memory fault must say so on stderr instead of dying silently.
//
// Both cases run in a forked child, since the observable behaviour *is* the
// process dying: the parent reads what the child wrote and how it died.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <util/base/signal_catcher.h>

#ifndef _WIN32
#  include <sys/wait.h>
#  include <unistd.h>
#  include <csignal>
#  include <string>

namespace
{
/** Run `crash` in a child with the reporter installed, returning what it wrote
 *  to stderr; `sig` receives the signal that killed it, or 0. */
std::string stderr_of_crash(void (*crash)(), int &sig)
{
  int fds[2];
  REQUIRE(pipe(fds) == 0);

  pid_t pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0)
  {
    close(fds[0]);
    dup2(fds[1], STDERR_FILENO);
    close(fds[1]);
    install_fatal_signal_reporter();
    crash();
    _exit(0); // Not reached: `crash` must fault.
  }

  close(fds[1]);
  std::string out;
  char buf[512];
  for (ssize_t n; (n = read(fds[0], buf, sizeof(buf))) > 0;)
    out.append(buf, n);
  close(fds[0]);

  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  sig = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
  return out;
}

void null_deref()
{
  *(volatile int *)nullptr = 1;
}

/** Eat the stack. The array is written and read so the frame cannot be
 *  optimised away, and the recursion is not a tail call. The depth bound is
 *  unreachable -- 16 KiB frames exhaust a default stack in a few hundred --
 *  and exists only because the compiler rejects provably infinite recursion. */
[[gnu::noinline]] int eat_stack(int depth)
{
  if (depth > 1000000)
    return 0;

  volatile char frame[16 * 1024];
  frame[0] = (char)depth;
  frame[sizeof(frame) - 1] = (char)depth;
  return frame[0] + eat_stack(depth + 1) + frame[sizeof(frame) - 1];
}

void stack_overflow()
{
  eat_stack(0);
}
} // namespace

TEST_CASE("a segfault reports itself on stderr", "[signal]")
{
  int sig = 0;
  std::string out = stderr_of_crash(null_deref, sig);

  CHECK(out.find("ESBMC caught SIGSEGV") != std::string::npos);
  CHECK(out.find("--segfault-handler") != std::string::npos);
  CHECK(sig == SIGSEGV);
}

TEST_CASE("an exhausted stack still reports itself", "[signal]")
{
  // The alternate signal stack is what makes this reportable: a handler cannot
  // run on the stack whose exhaustion raised the signal.
  int sig = 0;
  std::string out = stderr_of_crash(stack_overflow, sig);

  CHECK(out.find("ESBMC caught SIGSEGV") != std::string::npos);
  CHECK(sig == SIGSEGV);
}
#endif
