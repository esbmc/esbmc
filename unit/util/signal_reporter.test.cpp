// A fatal memory fault must say so on stderr instead of dying silently.
//
// Every case runs in a forked child, since the observable behaviour *is* the
// process dying: the parent reads what the child wrote and how it died.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <util/base/signal_catcher.h>

#ifndef _WIN32
#  include <sys/mman.h>
#  include <sys/resource.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  include <csignal>
#  include <cstdio>
#  include <cstdlib>
#  include <string>

namespace
{
/** Run `body` in a child with the reporter installed, returning what it wrote
 *  to stderr; `sig` receives the signal that killed it, or 0. */
std::string
stderr_of(void (*body)(), int &sig, const char *advice = "advice line\n")
{
  int fds[2];
  REQUIRE(pipe(fds) == 0);

  pid_t pid = fork();
  if (pid == 0)
  {
    // Bound the stack whatever the invoking shell's ulimit was: an ambient
    // `ulimit -s unlimited` otherwise lets the exhaustion case grow to
    // gigabytes and be killed by the OOM killer rather than SIGSEGV.
    struct rlimit rl = {1u << 20, 1u << 20};
    setrlimit(RLIMIT_STACK, &rl);

    if (dup2(fds[1], STDERR_FILENO) == -1)
      _exit(2);
    close(fds[0]);
    close(fds[1]);
    // Catch2 installs its own fatal-condition handler *and* its own alternate
    // signal stack, both of which the fork inherits. The reporter declines to
    // displace either, so clear them: otherwise the test exercises Catch2's
    // alternate stack rather than the one the reporter acquires.
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    stack_t off = {};
    off.ss_flags = SS_DISABLE;
    sigaltstack(&off, nullptr);
    install_fatal_signal_reporter(advice);
    body();
    _exit(0);
  }
  REQUIRE(pid > 0);

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
 *  unreachable under the child's 1 MiB limit, and exists only because the
 *  compiler rejects provably infinite recursion. */
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

/** Touch a mapping with nothing behind it, which is SIGBUS rather than
 *  SIGSEGV: the address is mapped, the file page it names does not exist. */
void truncated_mapping()
{
  char path[] = "/tmp/esbmc_sigbus_XXXXXX";
  int fd = mkstemp(path);
  if (fd == -1)
    _exit(3);
  unlink(path);
  if (ftruncate(fd, 4096) != 0)
    _exit(3);

  void *p = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED || ftruncate(fd, 0) != 0)
    _exit(3);
  *(volatile char *)p = 1;
}

void no_fault()
{
}

volatile sig_atomic_t other_handler_ran = 0;

void other_handler(int sig)
{
  static const char mine[] = "the other handler ran\n";
  other_handler_ran = 1;
  signal_safe_write(STDERR_FILENO, mine, sizeof(mine) - 1);
  ::signal(sig, SIG_DFL);
  ::raise(sig);
}

/** Claim SIGSEGV first, then ask for the reporter, then fault. */
void fault_under_foreign_handler()
{
  ::signal(SIGSEGV, other_handler);
  install_fatal_signal_reporter("advice line\n");
  null_deref();
}
} // namespace

TEST_CASE("a segfault reports itself on stderr", "[signal]")
{
  int sig = 0;
  std::string out = stderr_of(null_deref, sig);

  CHECK(out.find("ESBMC caught SIGSEGV") != std::string::npos);
  CHECK(out.find("advice line") != std::string::npos);
  CHECK(out.find("github.com/esbmc/esbmc/issues") != std::string::npos);
  CHECK(sig == SIGSEGV);
}

TEST_CASE("an exhausted stack still reports itself", "[signal]")
{
  // The alternate signal stack is what makes this reportable: a handler cannot
  // run on the stack whose exhaustion raised the signal.
  int sig = 0;
  std::string out = stderr_of(stack_overflow, sig);

  CHECK(out.find("ESBMC caught SIGSEGV") != std::string::npos);
  CHECK(sig == SIGSEGV);
}

TEST_CASE("a bus error names itself, not SIGSEGV", "[signal]")
{
  // Also the no-advice shape: a driver that offers no extra option, as c2goto
  // does, passes null and prints the report line alone.
  int sig = 0;
  std::string out = stderr_of(truncated_mapping, sig, nullptr);

  CHECK(out.find("ESBMC caught SIGBUS") != std::string::npos);
  CHECK(out.find("advice line") == std::string::npos);
  CHECK(out.find("SIGSEGV") == std::string::npos);
  CHECK(sig == SIGBUS);
}

TEST_CASE("a run that does not fault reports nothing", "[signal]")
{
  int sig = 0;
  std::string out = stderr_of(no_fault, sig);

  CHECK(out.empty());
  CHECK(sig == 0);
}

TEST_CASE("the reporter does not displace another SIGSEGV handler", "[signal]")
{
  // A sanitizer's handler prints far more than this one does, so an existing
  // disposition wins.
  int sig = 0;
  std::string out = stderr_of(fault_under_foreign_handler, sig);

  CHECK(out.find("the other handler ran") != std::string::npos);
  CHECK(out.find("ESBMC caught") == std::string::npos);
  CHECK(sig == SIGSEGV);
}
#endif
