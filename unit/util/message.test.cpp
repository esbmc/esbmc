/*******************************************************************\
Module: Unit tests for util/message/message.h
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <util/config/config.h>
#include <util/message/message.h>

#include <cstdio>
#include <string>
#include <vector>

#ifndef _WIN32
#  include <sys/wait.h>
#  include <unistd.h>
#endif

TEST_CASE("logln keeps the prefix on the message's own line", "[util][message]")
{
  FILE *f = tmpfile();
  REQUIRE(f != nullptr);

  messaget::statet st;
  st.verbosity = VerbosityLevel::Debug;
  st.out = f;
  st.logln(nullptr, VerbosityLevel::Warning, nullptr, 0, FMT_STRING("body"));

  rewind(f);
  char got[64] = {};
  REQUIRE(fgets(got, sizeof(got), f) != nullptr);
  REQUIRE(std::string(got) == "WARNING: body\n");
  fclose(f);
}

#ifndef _WIN32
/* A verdict spliced by a concurrently-writing child stops matching the
 * regressions' anchored regexes; --k-induction-parallel forks three children
 * onto one unbuffered descriptor. Every line must arrive whole. */
TEST_CASE("concurrent writers cannot splice a line", "[util][message]")
{
  static constexpr int writers = 8;
  static constexpr int lines_each = 400;

  int fds[2];
  REQUIRE(pipe(fds) == 0);

  std::vector<pid_t> kids;
  for (int w = 0; w < writers; w++)
  {
    pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0)
    {
      close(fds[0]);
      FILE *f = fdopen(fds[1], "w");
      setvbuf(f, nullptr, _IONBF, 0);
      messaget::statet st;
      st.verbosity = VerbosityLevel::Debug;
      st.out = f;
      for (int i = 0; i < lines_each; i++)
        st.logln(
          nullptr, VerbosityLevel::Result, nullptr, 0, FMT_STRING("VERDICT"));
      fflush(f);
      _exit(0);
    }
    kids.push_back(pid);
  }
  close(fds[1]);

  std::string all;
  char chunk[4096];
  for (ssize_t n; (n = read(fds[0], chunk, sizeof(chunk))) > 0;)
    all.append(chunk, n);
  close(fds[0]);

  for (pid_t pid : kids)
    waitpid(pid, nullptr, 0);

  size_t whole = 0;
  for (size_t pos = 0, nl; (nl = all.find('\n', pos)) != std::string::npos;
       pos = nl + 1)
  {
    REQUIRE(all.substr(pos, nl - pos) == "VERDICT");
    whole++;
  }
  REQUIRE(whole == writers * lines_each);
}
#endif
