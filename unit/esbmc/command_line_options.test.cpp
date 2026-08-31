/*******************************************************************
 Module: Command-line spec parser unit tests
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <esbmc/esbmc_parseoptions.h>

namespace
{
/// read_time_spec and read_mem_spec are protected members that use no state.
/// Deriving reaches them without changing their access in the driver.
struct spec_readert : esbmc_parseoptionst
{
  static const char *argv0;
  spec_readert() : esbmc_parseoptionst(1, &argv0)
  {
  }
  using esbmc_parseoptionst::read_mem_spec;
  using esbmc_parseoptionst::read_time_spec;
};
const char *spec_readert::argv0 = "esbmc";

/// One per process: constructing esbmc_parseoptionst parses all_cmd_options,
/// and a table is parsed exactly once. esbmc constructs one too.
spec_readert &reader()
{
  static spec_readert r;
  return r;
}

constexpr uint64_t kib = 1024;
constexpr uint64_t mib = 1024 * kib;
constexpr uint64_t gib = 1024 * mib;
} // namespace

TEST_CASE("a timeout with no suffix is seconds", "[command-line-options]")
{
  CHECK(reader().read_time_spec("0") == 0);
  CHECK(reader().read_time_spec("42") == 42);
}

TEST_CASE("every timeout suffix scales to seconds", "[command-line-options]")
{
  CHECK(reader().read_time_spec("90s") == 90);
  CHECK(reader().read_time_spec("2m") == 120);
  CHECK(reader().read_time_spec("3h") == 10800);
  CHECK(reader().read_time_spec("1d") == 86400);
}

TEST_CASE(
  "a memory limit with no suffix is megabytes",
  "[command-line-options]")
{
  CHECK(reader().read_mem_spec("1") == mib);
  CHECK(reader().read_mem_spec("512") == 512 * mib);
}

TEST_CASE("every memory suffix scales to bytes", "[command-line-options]")
{
  CHECK(reader().read_mem_spec("4096b") == 4096);
  CHECK(reader().read_mem_spec("64k") == 64 * kib);
  CHECK(reader().read_mem_spec("8m") == 8 * mib);
  CHECK(reader().read_mem_spec("2g") == 2 * gib);
}

TEST_CASE("a memory limit does not overflow 32 bits", "[command-line-options]")
{
  // 8g exceeds UINT32_MAX once scaled; the arithmetic is 64-bit throughout.
  CHECK(reader().read_mem_spec("8g") == 8ULL * gib);
  CHECK(reader().read_mem_spec("8g") > UINT32_MAX);
}

TEST_CASE("a timeout of days does not overflow", "[command-line-options]")
{
  CHECK(reader().read_time_spec("1000d") == 1000ULL * 86400);
}
