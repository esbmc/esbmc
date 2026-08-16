/*******************************************************************\
Module: Unit tests for filesystem_operations class
Author: Rafael Sá Menezes

\*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <util/base/filesystem.h>
#include <boost/filesystem.hpp>
#ifndef _WIN32
#  include <fcntl.h>
#  include <sys/file.h>
#  include <unistd.h>
#endif
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <thread>

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

TEST_CASE(
  "tmp file is created exclusively, not opened by name",
  "[core][util][filesystem]")
{
  // unique_path() only invents a name; it does not stake a claim on it.
  // Creating the file with fopen() therefore opened whatever already sat at
  // that path -- including a symlink planted between the two steps -- and
  // truncated it (CWE-377). Opening with O_EXCL|0600 instead is observable:
  // the temporary is not group/world accessible.
  auto file = file_operations::create_tmp_file("esbmc-test-%%%%");
  const std::string path = file.path();
  REQUIRE(boost::filesystem::is_regular_file(path));
  REQUIRE(
    !boost::filesystem::is_symlink(boost::filesystem::symlink_status(path)));

#ifndef _WIN32
  using boost::filesystem::perms;
  const perms p = boost::filesystem::status(path).permissions();
  REQUIRE((p & (perms::group_all | perms::others_all)) == perms::no_perms);
#endif
}

#ifndef _WIN32
TEST_CASE(
  "an already-taken tmp name is never clobbered",
  "[core][util][filesystem]")
{
  // A single '%' expands to one hex digit, so the format below has exactly 16
  // candidate names. Occupy 15 of them and the only name create_tmp_file() can
  // take is the sixteenth -- so the outcome is deterministic, and any run that
  // truncates a sentinel or returns an occupied name is a regression. fopen()
  // took the first name it drew, clobbering a sentinel 15 times out of 16.
  namespace fs = boost::filesystem;
  auto dir = file_operations::create_tmp_dir("esbmc-test-excl-%%%%");

  // create_tmp_file() resolves the temp directory per call, so TMPDIR confines
  // this test to `dir` and keeps it off the shared /tmp.
  const char *saved = getenv("TMPDIR");
  const std::string old_tmpdir = saved ? saved : "";
  setenv("TMPDIR", dir.path().c_str(), 1);

  const std::string free_name = "esbmc-test-slot-a";
  for (const char *d = "0123456789abcdef"; *d; ++d)
  {
    const fs::path occupied =
      fs::path(dir.path()) / ("esbmc-test-slot-" + std::string(1, *d));
    if (occupied.filename().string() == free_name)
      continue;
    std::ofstream(occupied.string()) << "sentinel";
  }
  REQUIRE(!fs::exists(fs::path(dir.path()) / free_name));

  {
    auto taken = file_operations::create_tmp_file("esbmc-test-slot-%");
    REQUIRE(fs::path(taken.path()).filename().string() == free_name);
  }

  for (const char *d = "0123456789abcdef"; *d; ++d)
  {
    const fs::path occupied =
      fs::path(dir.path()) / ("esbmc-test-slot-" + std::string(1, *d));
    if (occupied.filename().string() == free_name)
      continue;
    REQUIRE(fs::file_size(occupied) == 8);
  }

  if (saved)
    setenv("TMPDIR", old_tmpdir.c_str(), 1);
  else
    unsetenv("TMPDIR");
}
#endif

#ifndef _WIN32
// Returns true iff an exclusive BSD lock can be taken on `path`, i.e. nobody
// currently holds one. flock() conflicts across open file descriptions, so this
// sees the lock tmp_path holds even though we are the same process.
static bool can_lock_exclusively(const std::string &path)
{
  int fd = open(path.c_str(), O_RDONLY);
  REQUIRE(fd >= 0);
  bool free_to_lock = flock(fd, LOCK_EX | LOCK_NB) == 0;
  if (free_to_lock)
    flock(fd, LOCK_UN);
  close(fd);
  return free_to_lock;
}

TEST_CASE(
  "a temporary directory is flocked against systemd-tmpfiles",
  "[core][util][filesystem]")
{
  // systemd-tmpfiles ages /tmp by wall-clock time and would happily unlink a
  // long run's temporaries; it skips any path holding a BSD lock. Without the
  // lock this path would be lockable, so the check discriminates.
  std::string path;
  {
    auto dir = file_operations::create_tmp_dir("esbmc-test-lock-%%%%");
    path = dir.path();
    REQUIRE(!can_lock_exclusively(path));
  }
  REQUIRE(!boost::filesystem::exists(path));
}

TEST_CASE(
  "the temporary lock is released once the path is given up",
  "[core][util][filesystem]")
{
  // Keeping the descriptor open past our ownership would pin the inode and
  // exclude it from ageing for the rest of the process's life.
  std::string path;
  {
    auto dir = file_operations::create_tmp_dir("esbmc-test-lock-%%%%");
    path = dir.path();
    dir.keep(true);
  }
  REQUIRE(boost::filesystem::exists(path));
  REQUIRE(can_lock_exclusively(path));
  boost::filesystem::remove_all(path);
}

TEST_CASE("moving a temporary path moves its lock", "[core][util][filesystem]")
{
  // The move must hand the descriptor over, not re-acquire: a second flock()
  // from this process would succeed and leak the original descriptor.
  std::string path;
  {
    auto dir = file_operations::create_tmp_dir("esbmc-test-lock-%%%%");
    path = dir.path();
    auto moved = std::move(dir);
    REQUIRE(moved.path() == path);
    REQUIRE(!can_lock_exclusively(path));
  }
  REQUIRE(!boost::filesystem::exists(path));
}

TEST_CASE(
  "signal-safe cleanup removes a registered tree",
  "[core][util][filesystem]")
{
  // remove_registered_tmps_from_signal() is what the SIGALRM/SIGTERM handlers
  // call instead of cleanup_registered_tmps(), which allocates (#6201). It
  // must still remove a non-empty directory tree.
  const std::string root =
    file_operations::get_unique_tmp_path("esbmc-test-sigclean-%%%%-%%%%");
  boost::filesystem::create_directories(root + "/nested");
  std::ofstream(root + "/nested/file.txt") << "content";
  REQUIRE(boost::filesystem::exists(root + "/nested/file.txt"));

  file_operations::register_tmp_for_cleanup(root);
  file_operations::remove_registered_tmps_from_signal();

  // The removal runs in a forked child, so wait for it rather than racing it.
  for (int i = 0; i < 200 && boost::filesystem::exists(root); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  REQUIRE(!boost::filesystem::exists(root));

  file_operations::cleanup_registered_tmps();
}
#endif
