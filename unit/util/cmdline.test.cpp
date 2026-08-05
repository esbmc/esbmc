#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <util/config/cmdline.h>

#include <cstdlib>
#include <string>
#include <vector>

// Reaching config-file location resolution is what matters here: parse()
// expands the default "~/.config/esbmc.toml" before it can consult HOME.
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
/// Restores HOME on scope exit so one section cannot leak into the next.
struct scoped_home
{
  bool had_home;
  std::string saved;

  explicit scoped_home(const char *value)
  {
    const char *current = std::getenv("HOME");
    had_home = current != nullptr;
    if (had_home)
      saved = current;
    if (value)
      setenv("HOME", value, 1);
    else
      unsetenv("HOME");
  }

  ~scoped_home()
  {
    if (had_home)
      setenv("HOME", saved.c_str(), 1);
    else
      unsetenv("HOME");
  }
};
} // namespace

TEST_CASE("cmdline parses with HOME unset", "[core][util][cmdline]")
{
  const char *argv[] = {"esbmc", "main.c"};

  SECTION("HOME absent from the environment")
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

  SECTION("HOME set but empty")
  {
    scoped_home empty_home("");
    const std::vector<group_opt_templ> opts = make_test_options();
    cmdlinet cmdline;
    REQUIRE_FALSE(cmdline.parse(2, argv, opts.data()));
    REQUIRE(cmdline.args[0] == "main.c");
  }

  SECTION("HOME set")
  {
    scoped_home a_home("/tmp");
    const std::vector<group_opt_templ> opts = make_test_options();
    cmdlinet cmdline;
    REQUIRE_FALSE(cmdline.parse(2, argv, opts.data()));
    REQUIRE(cmdline.args[0] == "main.c");
  }
}
