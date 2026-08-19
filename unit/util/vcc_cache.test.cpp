#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <util/ssa/vcc_cache.h>

#include <filesystem>
#include <fstream>

namespace
{
std::string scratch_dir(const std::string &tag)
{
  const auto dir =
    std::filesystem::temp_directory_path() / ("esbmc-vcc-cache-test-" + tag);
  std::filesystem::remove_all(dir);
  return dir.string();
}

optionst with_option(const std::string &name, const std::string &value)
{
  optionst o;
  o.set_option(name, value);
  return o;
}
} // namespace

TEST_CASE("a recorded cone is proved, an unrecorded one is not", "[vcc_cache]")
{
  const std::string dir = scratch_dir("roundtrip");
  optionst options;
  vcc_cachet cache(dir, options);

  const std::string cone = "step 1\nguard\ncond\n";

  REQUIRE_FALSE(cache.proved(cone));
  cache.record(cone);
  REQUIRE(cache.proved(cone));
  REQUIRE_FALSE(cache.proved(cone + "extra\n"));

  REQUIRE(cache.hits() == 1);
  REQUIRE(cache.misses() == 2);

  std::filesystem::remove_all(dir);
}

TEST_CASE("recording twice is idempotent", "[vcc_cache]")
{
  const std::string dir = scratch_dir("idempotent");
  optionst options;
  vcc_cachet cache(dir, options);

  const std::string cone = "step 1\nrepeat\n";
  cache.record(cone);
  cache.record(cone);
  REQUIRE(cache.proved(cone));

  size_t entries = 0;
  for (const auto &e : std::filesystem::recursive_directory_iterator(dir))
    if (e.path().extension() == ".vcc")
      ++entries;
  REQUIRE(entries == 1);

  std::filesystem::remove_all(dir);
}

TEST_CASE("a different key is a miss", "[vcc_cache]")
{
  const std::string dir = scratch_dir("distinct");
  optionst options;
  vcc_cachet cache(dir, options);

  // Entries are named by a 128-bit key and hold nothing; presence is the
  // proof, so distinctness is the whole safety argument.
  cache.record("0123456789abcdef0123456789abcdef");
  REQUIRE(cache.proved("0123456789abcdef0123456789abcdef"));
  REQUIRE_FALSE(cache.proved("0123456789abcdef0123456789abcdee"));

  std::filesystem::remove_all(dir);
}

TEST_CASE("a proof does not carry across option sets", "[vcc_cache]")
{
  const std::string dir = scratch_dir("context");
  const std::string cone = "step 1\nshared\n";

  vcc_cachet floats(dir, with_option("floatbv", "1"));
  floats.record(cone);
  REQUIRE(floats.proved(cone));

  vcc_cachet fixed(dir, with_option("fixedbv", "1"));
  REQUIRE_FALSE(fixed.proved(cone));

  std::filesystem::remove_all(dir);
}

TEST_CASE("the cache's own controls are not part of the key", "[vcc_cache]")
{
  const std::string dir = scratch_dir("controls");
  const std::string cone = "step 1\ncontrols\n";

  vcc_cachet plain(dir, with_option("vcc-cache", dir));
  plain.record(cone);

  // --vcc-cache-verify must read what a plain run wrote, or it can never
  // check anything.
  optionst verifying;
  verifying.set_option("vcc-cache", dir);
  verifying.set_option("vcc-cache-verify", true);
  REQUIRE(vcc_cachet(dir, verifying).proved(cone));

  std::filesystem::remove_all(dir);
}

TEST_CASE("a cache directory that cannot be created is inert", "[vcc_cache]")
{
  const std::string path = scratch_dir("not-a-directory");
  std::ofstream(path) << "occupied\n";

  optionst options;
  vcc_cachet cache(path, options);

  const std::string cone = "step 1\nblocked\n";
  // Nothing can be written, so the run must go on solving rather than fail.
  cache.record(cone);
  REQUIRE_FALSE(cache.proved(cone));
  REQUIRE(cache.hits() == 0);

  std::filesystem::remove(path);
}

TEST_CASE("a failed rename leaves no temporary behind", "[vcc_cache]")
{
  const std::string dir = scratch_dir("rename-fails");
  optionst options;
  vcc_cachet cache(dir, options);

  const std::string cone = "step 1\nrenamed\n";
  cache.record(cone);

  // Make the entry's own path un-renamable-onto by turning it into a
  // directory; the temp file must not be left lying in the store.
  std::filesystem::path entry;
  for (const auto &e : std::filesystem::recursive_directory_iterator(dir))
    if (e.path().extension() == ".vcc")
      entry = e.path();
  REQUIRE_FALSE(entry.empty());
  std::filesystem::remove(entry);
  std::filesystem::create_directory(entry);

  cache.record(cone);

  size_t leftovers = 0;
  for (const auto &e : std::filesystem::recursive_directory_iterator(dir))
    if (e.path().filename().string().find(".tmp") != std::string::npos)
      ++leftovers;
  REQUIRE(leftovers == 0);

  std::filesystem::remove_all(dir);
}
