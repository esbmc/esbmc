#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <util/config/config.h>
#include <util/ssa/proof_cache.h>

#include <filesystem>
#include <fstream>

namespace
{
std::string scratch_dir(const std::string &tag)
{
  const auto dir =
    std::filesystem::temp_directory_path() / ("esbmc-proof-cache-test-" + tag);
  std::filesystem::remove_all(dir);
  return dir.string();
}

optionst with_option(const std::string &name, const std::string &value)
{
  optionst o;
  o.set_option(name, value);
  return o;
}

/// Stands in for the build ID linked into esbmc; any fixed string works, since
/// what matters is only that two runs agree on it.
const std::string build = "esbmc built from cafef00d";
} // namespace

TEST_CASE(
  "a recorded cone is proved, an unrecorded one is not",
  "[proof_cache]")
{
  const std::string dir = scratch_dir("roundtrip");
  optionst options;
  proof_cachet cache(dir, options, build);

  const std::string cone = "step 1\nguard\ncond\n";

  REQUIRE_FALSE(cache.proved(cone));
  cache.record(cone);
  REQUIRE(cache.proved(cone));
  REQUIRE_FALSE(cache.proved(cone + "extra\n"));

  REQUIRE(cache.hits() == 1);
  REQUIRE(cache.misses() == 2);

  std::filesystem::remove_all(dir);
}

TEST_CASE("recording twice is idempotent", "[proof_cache]")
{
  const std::string dir = scratch_dir("idempotent");
  optionst options;
  proof_cachet cache(dir, options, build);

  const std::string cone = "step 1\nrepeat\n";
  cache.record(cone);
  cache.record(cone);
  REQUIRE(cache.proved(cone));

  size_t entries = 0;
  for (const auto &e : std::filesystem::recursive_directory_iterator(dir))
    if (e.path().extension() == ".proof")
      ++entries;
  REQUIRE(entries == 1);

  std::filesystem::remove_all(dir);
}

TEST_CASE("a different key is a miss", "[proof_cache]")
{
  const std::string dir = scratch_dir("distinct");
  optionst options;
  proof_cachet cache(dir, options, build);

  // Entries are named by a 128-bit key and hold nothing; presence is the
  // proof, so distinctness is the whole safety argument.
  cache.record("0123456789abcdef0123456789abcdef");
  REQUIRE(cache.proved("0123456789abcdef0123456789abcdef"));
  REQUIRE_FALSE(cache.proved("0123456789abcdef0123456789abcdee"));

  std::filesystem::remove_all(dir);
}

TEST_CASE("a proof does not carry across option sets", "[proof_cache]")
{
  const std::string dir = scratch_dir("context");
  const std::string cone = "step 1\nshared\n";

  proof_cachet floats(dir, with_option("floatbv", "1"), build);
  floats.record(cone);
  REQUIRE(floats.proved(cone));

  proof_cachet fixed(dir, with_option("fixedbv", "1"), build);
  REQUIRE_FALSE(fixed.proved(cone));

  std::filesystem::remove_all(dir);
}

TEST_CASE("the cache's own controls are not part of the key", "[proof_cache]")
{
  const std::string dir = scratch_dir("controls");
  const std::string cone = "step 1\ncontrols\n";

  proof_cachet plain(dir, with_option("proof-cache", dir), build);
  plain.record(cone);

  // --proof-cache-verify must read what a plain run wrote, or it can never
  // check anything.
  optionst verifying;
  verifying.set_option("proof-cache", dir);
  verifying.set_option("proof-cache-verify", true);
  REQUIRE(proof_cachet(dir, verifying, build).proved(cone));

  std::filesystem::remove_all(dir);
}

TEST_CASE("a cache directory that cannot be created is inert", "[proof_cache]")
{
  const std::string path = scratch_dir("not-a-directory");
  std::ofstream(path) << "occupied\n";

  optionst options;
  proof_cachet cache(path, options, build);

  const std::string cone = "step 1\nblocked\n";
  // Nothing can be written, so the run must go on solving rather than fail.
  cache.record(cone);
  REQUIRE_FALSE(cache.proved(cone));
  REQUIRE(cache.hits() == 0);

  std::filesystem::remove(path);
}

TEST_CASE("a failed rename leaves no temporary behind", "[proof_cache]")
{
  const std::string dir = scratch_dir("rename-fails");
  optionst options;
  proof_cachet cache(dir, options, build);

  const std::string cone = "step 1\nrenamed\n";
  cache.record(cone);

  // Make the entry's own path un-renamable-onto by turning it into a
  // directory; the temp file must not be left lying in the store.
  std::filesystem::path entry;
  for (const auto &e : std::filesystem::recursive_directory_iterator(dir))
    if (e.path().extension() == ".proof")
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

TEST_CASE("a proof does not carry across builds", "[proof_cache]")
{
  const std::string dir = scratch_dir("build");
  const std::string cone = "step 1\nbuilt\n";
  optionst options;

  proof_cachet(dir, options, "esbmc built from cafef00d").record(cone);
  REQUIRE(proof_cachet(dir, options, "esbmc built from cafef00d").proved(cone));
  REQUIRE_FALSE(
    proof_cachet(dir, options, "esbmc built from 0ddba11").proved(cone));

  std::filesystem::remove_all(dir);
}

TEST_CASE(
  "an option that reaches only the report is not part of the key",
  "[proof_cache]")
{
  const std::string dir = scratch_dir("presentation");
  const std::string cone = "step 1\npresentation\n";

  // --color resolves from isatty(), so an interactive run and a piped one
  // disagree on it without ever disagreeing on what is verified.
  proof_cachet(dir, with_option("color", "1"), build).record(cone);
  REQUIRE(proof_cachet(dir, with_option("color", "0"), build).proved(cone));

  for (const char *opt :
       {"ascii-report",
        "cex-output",
        "file-output",
        "log-message",
        "quiet",
        "verbosity"})
    REQUIRE(proof_cachet(dir, with_option(opt, "1"), build).proved(cone));

  std::filesystem::remove_all(dir);
}

TEST_CASE(
  "an option that changes what is verified is part of the key",
  "[proof_cache]")
{
  const std::string dir = scratch_dir("semantic");
  const std::string cone = "step 1\nsemantic\n";

  proof_cachet(dir, optionst(), build).record(cone);
  for (const char *opt :
       {"unwind",
        "no-pointer-check",
        "overflow-check",
        "no-slice",
        "unsigned-overflow-check"})
    REQUIRE_FALSE(proof_cachet(dir, with_option(opt, "1"), build).proved(cone));

  std::filesystem::remove_all(dir);
}

TEST_CASE(
  "the modes a stored proof would not be sound in are named",
  "[proof_cache]")
{
  REQUIRE(proof_cache_inactive_reason(optionst()).empty());

  for (const char *opt :
       {"coverage-measurement", "dead-code-check", "ltl", "smt-during-symex"})
    REQUIRE_FALSE(proof_cache_inactive_reason(with_option(opt, "1")).empty());

  // The option is written unconditionally, so a run that does not measure
  // coverage carries it as false.
  REQUIRE(proof_cache_inactive_reason(with_option("coverage-measurement", "0"))
            .empty());
}

TEST_CASE("a build ID that names one build is the identity", "[proof_cache]")
{
  const std::string clean = "ESBMC built from cafef00d 2026-01-01 by a@b";
  REQUIRE(proof_cache_build_identity(clean) == clean);

  // A dirty tree, or a build with no commit to name, describes a class of
  // builds; the identity must then say more than the ID does.
  for (const std::string &id :
       {clean + " (dirty tree)",
        std::string("ESBMC built from no-hash by a@b")})
  {
    const std::string identity = proof_cache_build_identity(id);
    REQUIRE(identity != id);
    REQUIRE(identity.rfind(id, 0) == 0);
  }
}

TEST_CASE("only a counterexample contradicts a stored proof", "[proof_cache]")
{
  // Reached only under --proof-cache-verify, which solves a claim the cache
  // already answered. A refutation means the cache is wrong about it.
  REQUIRE(proof_cache_contradicted(true, true));

  // Everything else leaves the stored proof standing: a claim that was not
  // hit says nothing about the cache, and a solver that neither proved nor
  // refuted (an error, an SMT-LIB-only emission, a vacuous discharge) did not
  // check it.
  REQUIRE_FALSE(proof_cache_contradicted(true, false));
  REQUIRE_FALSE(proof_cache_contradicted(false, true));
  REQUIRE_FALSE(proof_cache_contradicted(false, false));
}

TEST_CASE(
  "every value of a repeatable option is part of the key",
  "[proof_cache]")
{
  const std::string dir = scratch_dir("repeatable");
  const std::string cone = "step 1\nrepeatable\n";

  // set_option overwrites, so `-DA=1 -DB=1` and `-DB=1` leave the same entry
  // in option_map. They are different verifications and must key apart.
  optionst both = with_option("D", "B=1");
  both.option_values["D"] = {"A=1", "B=1"};
  optionst last = with_option("D", "B=1");
  last.option_values["D"] = {"B=1"};

  proof_cachet(dir, both, build).record(cone);
  REQUIRE(proof_cachet(dir, both, build).proved(cone));
  REQUIRE_FALSE(proof_cachet(dir, last, build).proved(cone));

  std::filesystem::remove_all(dir);
}

TEST_CASE("the target and the data model are part of the key", "[proof_cache]")
{
  // Neither is in the option set: absent a --32 or --i386-linux style flag
  // ESBMC takes both from the machine it is running on, so two hosts sharing
  // a cache directory would otherwise agree on every key.
  const std::string base = proof_cache_context(optionst(), build);
  const configt saved = config;

  config.ansi_c.target.arch = "riscv64";
  REQUIRE(proof_cache_context(optionst(), build) != base);
  config = saved;

  config.ansi_c.target.os = "macos";
  REQUIRE(proof_cache_context(optionst(), build) != base);
  config = saved;

  config.ansi_c.long_double_width = 64;
  REQUIRE(proof_cache_context(optionst(), build) != base);
  config = saved;

  config.ansi_c.wchar_t_width = 16;
  REQUIRE(proof_cache_context(optionst(), build) != base);
  config = saved;

  config.language.cpp_std = cxx_stdt::cpp20;
  REQUIRE(proof_cache_context(optionst(), build) != base);
  config = saved;

  REQUIRE(proof_cache_context(optionst(), build) == base);
}

TEST_CASE("a proof does not carry across solver sets", "[proof_cache]")
{
  // ESBMC_AVAILABLE_SOLVERS is compiled in, so this cannot be varied from a
  // test; what it can pin is that the context names it at all, which is what
  // keeps two builds with different backends apart.
  const std::string ctx = proof_cache_context(optionst(), build);
  REQUIRE(ctx.find("solvers ") != std::string::npos);
  REQUIRE(ctx.find("build " + build) != std::string::npos);
}
