#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <util/config/cmdline.h>

#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <stdlib.h> // _putenv_s
// Mirrors HOME_ENV_NAME in src/util/config/cmdline.cpp; keep the two in step.
#  define HOME_ENV_NAME "USERPROFILE"
#else
#  define HOME_ENV_NAME "HOME"
#endif

// Reaching config-file location resolution is what matters here: parse()
// expands the default config path before it can consult the home variable.
// Built afresh per call: parse() hands each value_semantic to boost, which
// takes ownership, so one table cannot serve two parses.
static std::vector<group_opt_templ> make_test_options()
{
  return {
    {"Basic Usage",
     {{"input-file",
       boost::program_options::value<std::vector<std::string>>(),
       "source file names"}}},
    {"end", {{"", NULL, "end of options"}}},
    {"Hidden Options", {{"", NULL, ""}}}};
}

namespace
{
/// Sets the home variable, or removes it from the environment when `value` is
/// null.
void put_home(const char *value)
{
#ifdef _WIN32
  // The MSVC CRT has no setenv/unsetenv; _putenv_s removes a variable when
  // given an empty value, and rejects a null one.
  _putenv_s(HOME_ENV_NAME, value ? value : "");
#else
  if (value)
    setenv(HOME_ENV_NAME, value, 1);
  else
    unsetenv(HOME_ENV_NAME);
#endif
}

/// Restores the home variable on scope exit so one section cannot leak into
/// the next.
struct scoped_home
{
  bool had_home;
  std::string saved;

  explicit scoped_home(const char *value)
  {
    const char *current = std::getenv(HOME_ENV_NAME);
    had_home = current != nullptr;
    if (had_home)
      saved = current;
    put_home(value);
  }

  ~scoped_home()
  {
    put_home(had_home ? saved.c_str() : nullptr);
  }
};
} // namespace

TEST_CASE(
  "cmdline parses with the home variable unset",
  "[core][util][cmdline]")
{
  const char *argv[] = {"esbmc", "main.c"};

  SECTION("home variable absent from the environment")
  {
    // Constructing std::optional<std::string> from getenv's null return is UB
    // and aborted or crashed the process before any work was done (#6238).
    scoped_home no_home(nullptr);
    const std::vector<group_opt_templ> opts = make_test_options();
    cmdlinet cmdline;
    REQUIRE_FALSE(cmdline.parse(2, argv, opts.data()));
    REQUIRE(cmdline.args.size() == 1);
    REQUIRE(cmdline.args[0] == "main.c");
  }

#ifndef _WIN32
  // Windows cannot hold an empty-but-set variable -- _putenv_s deletes one
  // assigned "" -- so there this case would just repeat the one above.
  SECTION("home variable set but empty")
  {
    scoped_home empty_home("");
    const std::vector<group_opt_templ> opts = make_test_options();
    cmdlinet cmdline;
    REQUIRE_FALSE(cmdline.parse(2, argv, opts.data()));
    REQUIRE(cmdline.args[0] == "main.c");
  }
#endif

  SECTION("home variable set")
  {
    scoped_home a_home("/tmp");
    const std::vector<group_opt_templ> opts = make_test_options();
    cmdlinet cmdline;
    REQUIRE_FALSE(cmdline.parse(2, argv, opts.data()));
    REQUIRE(cmdline.args[0] == "main.c");
  }
}
