/*******************************************************************\
Module: Unit tests for filesystem_operations class
Author: Rafael Sá Menezes

\*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <util/filesystem.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <set>

#ifndef _WIN32
#  include <cstdlib>
#  include <sys/stat.h>
#  include <unistd.h>

namespace
{
/* cached_extract_dir() resolves its root once per process, so redirect it at
 * static-init time — before any test can touch it — and take the throwaway
 * root down again on exit. ESBMC_CACHE_DIR overrides every build variant, so
 * the tests do not have to care where the cache would normally land. */
struct scoped_cache_root
{
  std::string path;

  scoped_cache_root()
    : path(boost::filesystem::unique_path(
             boost::filesystem::temp_directory_path() /
             "esbmc-cache-test-%%%%-%%%%")
             .string())
  {
    setenv("ESBMC_CACHE_DIR", path.c_str(), 1);
  }

  ~scoped_cache_root()
  {
    boost::system::error_code ec;
    boost::filesystem::remove_all(path, ec);
  }
};

const scoped_cache_root cache_root;
} // namespace
#endif

TEST_CASE(
  "tmp path should be unique between two runs",
  "[core][util][filesystem]")
{
  const char *format = "esbmc-test-%%%%";
  auto first = file_operations::get_unique_tmp_path(format);
  auto second = file_operations::get_unique_tmp_path(format);
  REQUIRE(first != second);
}

TEST_CASE(
  "tmp folder should be unique between two runs",
  "[core][util][filesystem]")
{
  const char *format = "esbmc-test-%%%%";
  auto first = file_operations::create_tmp_dir(format);
  auto second = file_operations::create_tmp_dir(format);
  REQUIRE(first.path() != second.path());
}

TEST_CASE(
  "tmp file should be unique between two runs",
  "[core][util][filesystem]")
{
  const char *format = "esbmc-test-%%%%";
  auto first = file_operations::create_tmp_file(format);
  auto second = file_operations::create_tmp_file(format);
  REQUIRE(first.path() != second.path());
}

TEST_CASE("tmp dir is dir and should be removed", "[core][util][filesystem]")
{
  const char *format = "esbmc-test-%%%%";
  std::string path;
  {
    auto dir = file_operations::create_tmp_dir(format);
    path = dir.path();
    REQUIRE(boost::filesystem::is_directory(path));
  }
  REQUIRE(!boost::filesystem::exists(path));
}

TEST_CASE("tmp file is file and should be removed", "[core][util][filesystem]")
{
  const char *format = "esbmc-test-%%%%";
  std::string path;
  {
    auto file = file_operations::create_tmp_file(format);
    path = file.path();
    REQUIRE(boost::filesystem::is_regular_file(path));
  }
  REQUIRE(!boost::filesystem::exists(path));
}

TEST_CASE(
  "tmp_path destructor tolerates an already-removed path",
  "[core][util][filesystem]")
{
  // create_tmp_dir() also registers the path with register_tmp_for_cleanup(),
  // so cleanup_registered_tmps() (run from the signal handler before exit()
  // triggers static/RAII destructors) can remove the directory before the
  // tmp_path destructor runs. Pre-fix, the destructor asserted removed >= 1
  // and aborted (SIGABRT) on SIGTERM/SIGINT, e.g. a benchexec timeout. The
  // destructor must instead tolerate the directory already being gone.
  const char *format = "esbmc-test-%%%%";
  std::string path;
  {
    auto dir = file_operations::create_tmp_dir(format);
    path = dir.path();
    REQUIRE(boost::filesystem::is_directory(path));
    // Simulate the registered-cleanup / signal-handler removing it first.
    boost::filesystem::remove_all(path);
    REQUIRE(!boost::filesystem::exists(path));
    // dir's destructor runs at end of scope: must not abort.
  }
  REQUIRE(!boost::filesystem::exists(path));
}

#ifndef _WIN32
static void write_payload(const std::string &dir)
{
  std::ofstream(dir + "/payload").write("hello", 5);
}

/* The root cached_extract_dir() must be using, given the redirect above. Tests
 * assert against this rather than deriving it from an entry's parent: on the
 * fallback path a returned entry's parent is $TMPDIR, and treating that as the
 * cache root turns a silent fallback into a nonsensical assertion about /tmp. */
static std::string expected_root()
{
  return cache_root.path;
}

TEST_CASE(
  "cached_extract_dir extracts once and reuses the entry",
  "[core][util][filesystem]")
{
  unsigned extractions = 0;
  auto extract = [&extractions](const std::string &dir) {
    ++extractions;
    write_payload(dir);
  };

  auto first =
    file_operations::cached_extract_dir("unit-test", "key-A", extract);
  REQUIRE(extractions == 1);
  REQUIRE(boost::filesystem::is_regular_file(first.path() + "/payload"));

  auto second =
    file_operations::cached_extract_dir("unit-test", "key-A", extract);
  REQUIRE(second.path() == first.path());
  REQUIRE(extractions == 1);
}

TEST_CASE(
  "cached_extract_dir separates distinct content keys",
  "[core][util][filesystem]")
{
  auto a =
    file_operations::cached_extract_dir("unit-test", "key-B", write_payload);
  auto b =
    file_operations::cached_extract_dir("unit-test", "key-C", write_payload);
  REQUIRE(a.path() != b.path());
}

TEST_CASE(
  "cached entry outlives the returned handle",
  "[core][util][filesystem]")
{
  std::string path;
  {
    auto dir =
      file_operations::cached_extract_dir("unit-test", "key-D", write_payload);
    path = dir.path();
    REQUIRE(boost::filesystem::is_directory(path));
    // Unlike create_tmp_dir(), the destructor must leave the entry in place.
  }
  REQUIRE(boost::filesystem::is_directory(path));
}

TEST_CASE(
  "cached_extract_dir publishes without leaving staging dirs",
  "[core][util][filesystem]")
{
  auto dir =
    file_operations::cached_extract_dir("unit-test", "key-E", write_payload);
  REQUIRE(dir.path().rfind(expected_root() + "/", 0) == 0);

  // Staging directories are named ".<name>.tmp-*" and must be gone once the
  // entry is published, whether we won the rename or lost it.
  unsigned staging = 0;
  for (const auto &e : boost::filesystem::directory_iterator(expected_root()))
    if (e.path().filename().string().rfind(".unit-test.tmp-", 0) == 0)
      ++staging;
  REQUIRE(staging == 0);
}

TEST_CASE(
  "a throwing extract is not published and falls back",
  "[core][util][filesystem]")
{
  // A partial extraction (e.g. ENOSPC) must never become a cache entry that a
  // later run trusts; the call degrades to a removed-on-exit temp dir instead.
  // Throw only on the staging attempt so the per-run fallback then succeeds.
  unsigned calls = 0;
  auto boom = [&calls](const std::string &dir) {
    write_payload(dir);
    if (++calls == 1)
      throw std::runtime_error("simulated write failure");
  };

  // Tolerate a not-yet-created root: this case may run as its own process
  // (catch_discover_tests), so nothing has materialised the cache root before
  // the first snapshot.
  auto entries = [] {
    std::set<std::string> s;
    boost::system::error_code ec;
    for (boost::filesystem::directory_iterator it(expected_root(), ec), end;
         it != end;
         it.increment(ec))
      s.insert(it->path().filename().string());
    return s;
  };
  const std::set<std::string> before = entries();

  std::string path;
  {
    auto dir = file_operations::cached_extract_dir("unit-test", "key-G", boom);
    path = dir.path();
    // Fell back outside the cache root, and is a temp dir (removed on scope
    // exit), not a published entry.
    REQUIRE(path.rfind(expected_root() + "/", 0) != 0);
    REQUIRE(boost::filesystem::is_directory(path));
  }
  REQUIRE(!boost::filesystem::exists(path));

  // The failed extraction published nothing new under the cache root.
  REQUIRE(entries() == before);
}

TEST_CASE(
  "cache root is owned by the user at mode 0700",
  "[core][util][filesystem]")
{
  // The root has a predictable, shared name, so it is created private to us:
  // that is what keeps our reusable entries from being confused with another
  // user's, or with a stale root left by an earlier build.
  auto dir =
    file_operations::cached_extract_dir("unit-test", "key-F", write_payload);
  REQUIRE(dir.path().rfind(expected_root() + "/", 0) == 0);

  struct stat st;
  REQUIRE(lstat(expected_root().c_str(), &st) == 0);
  REQUIRE(S_ISDIR(st.st_mode));
  REQUIRE(st.st_uid == getuid());
  REQUIRE((st.st_mode & 077) == 0);
}
#endif
