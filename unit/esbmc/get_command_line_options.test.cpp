/*******************************************************************
 Module: get_command_line_options unit tests
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <esbmc/esbmc_parseoptions.h>

namespace
{
struct option_readert : esbmc_parseoptionst
{
  option_readert(int argc, const char **argv) : esbmc_parseoptionst(argc, argv)
  {
  }
  using esbmc_parseoptionst::get_command_line_options;
};

const char *driver_argv[] = {
  "esbmc",
  "--context-bound",
  "3",
  "--max-context-bound",
  "7",
  "--deadlock-check",
  "--compact-trace",
  "--base-case",
  "--fixedbv",
  "--smt-symex-guard",
  "main.c"};

/// esbmc_parseoptionst parses all_cmd_options on construction, and a table is
/// parsed once per process, so the whole binary shares one command line.
const optionst &driver_options()
{
  static const optionst options = [] {
    option_readert reader(
      static_cast<int>(std::size(driver_argv)), driver_argv);
    optionst built;
    reader.get_command_line_options(built);
    return built;
  }();
  return options;
}
} // namespace

TEST_CASE("a bound given on the command line reaches options", "[driver]")
{
  CHECK(driver_options().get_option("context-bound") == "3");
  CHECK(driver_options().get_option("max-context-bound") == "7");
}

TEST_CASE("an unset bound falls back to its default", "[driver]")
{
  // --incremental-context-bound was not given, so it stays off rather than
  // inheriting the bound above.
  CHECK_FALSE(driver_options().get_bool_option("incremental-context-bound"));
}

TEST_CASE("deadlock checking turns atomicity checking off", "[driver]")
{
  CHECK(driver_options().get_bool_option("deadlock-check"));
  CHECK_FALSE(driver_options().get_bool_option("atomicity-check"));
}

TEST_CASE("a compact trace keeps the slicer away from it", "[driver]")
{
  CHECK(driver_options().get_bool_option("no-slice"));
}

TEST_CASE("fixed-point encoding displaces floating point", "[driver]")
{
  CHECK(driver_options().get_bool_option("fixedbv"));
  CHECK_FALSE(driver_options().get_bool_option("floatbv"));
}

TEST_CASE("the base case runs without unwinding assertions", "[driver]")
{
  CHECK(driver_options().get_bool_option("base-case"));
  CHECK(driver_options().get_bool_option("no-unwinding-assertions"));
  CHECK_FALSE(driver_options().get_bool_option("partial-loops"));
}

TEST_CASE("an SMT symex guard pulls in SMT during symex", "[driver]")
{
  CHECK(driver_options().get_bool_option("smt-during-symex"));
}

TEST_CASE("integer encoding is left off without --ir", "[driver]")
{
  CHECK_FALSE(driver_options().get_bool_option("int-encoding"));
}
