/*******************************************************************\
Module: Unit tests for filesystem_operations class
Author: Rafael Sá Menezes

\*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <util/base/filesystem.h>
#include <boost/filesystem.hpp>

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

static const char bundled_bytes[] = {'h', 'e', 'l', 'l', 'o'};

TEST_CASE(
  "bundled file_data borrows without copying",
  "[core][util][filesystem]")
{
  auto f =
    file_operations::file_data::bundled(bundled_bytes, sizeof(bundled_bytes));
  // Aliases .rodata rather than copying it.
  REQUIRE(f.view().data() == bundled_bytes);
  REQUIRE(f.size() == 5);
  REQUIRE(f.is_bundled());
  REQUIRE(f.view() == "hello");
}

TEST_CASE("owned file_data holds its contents", "[core][util][filesystem]")
{
  auto f = file_operations::file_data::owned("from disk");
  REQUIRE(!f.is_bundled());
  REQUIRE(f.view() == "from disk");
  REQUIRE(f.size() == 9);
}

TEST_CASE(
  "moving owned file_data keeps the view valid",
  "[core][util][filesystem]")
{
  // A short string is stored inline (SSO), so moving it relocates the bytes.
  // A cached view would be left pointing into the moved-from object.
  auto first = file_operations::file_data::owned("abc");
  const char *before = first.view().data();
  auto second = std::move(first);
  REQUIRE(!second.is_bundled());
  REQUIRE(second.view() == "abc");
  REQUIRE(second.view().data() != before);
}

TEST_CASE(
  "moving bundled file_data keeps aliasing .rodata",
  "[core][util][filesystem]")
{
  auto first =
    file_operations::file_data::bundled(bundled_bytes, sizeof(bundled_bytes));
  auto second = std::move(first);
  REQUIRE(second.is_bundled());
  REQUIRE(second.view().data() == bundled_bytes);
}

TEST_CASE(
  "assigning file_data tracks the new contents",
  "[core][util][filesystem]")
{
  file_operations::file_data f;
  f = file_operations::file_data::owned("xy");
  REQUIRE(!f.is_bundled());
  REQUIRE(f.view() == "xy");

  f = file_operations::file_data::bundled(bundled_bytes, sizeof(bundled_bytes));
  REQUIRE(f.is_bundled());
  REQUIRE(f.view().data() == bundled_bytes);
}

TEST_CASE(
  "an empty file read from disk is not bundled",
  "[core][util][filesystem]")
{
  // A zero-byte file on disk is legitimate and still not a bundled file.
  auto f = file_operations::file_data::owned("");
  REQUIRE(!f.is_bundled());
  REQUIRE(f.size() == 0);
  REQUIRE(f.view().empty());
}

static const char hdr_bytes[] = "int a;";
static const char src_bytes[] = "int b;";

// Registration is process-wide, so each test uses its own subtree.
static std::string
register_tree(const std::string &suite, file_operations::filesystemt &fs)
{
  std::string root = std::string(file_operations::ESBMC_VFS_ROOT) + "/" + suite;
  fs.add_bundled(root + "/include/a.h", hdr_bytes, sizeof(hdr_bytes) - 1);
  fs.add_bundled(root + "/lib/sub/b.c", src_bytes, sizeof(src_bytes) - 1);
  return root;
}

TEST_CASE("bundled files resolve by path", "[core][util][filesystem]")
{
  auto &fs = file_operations::filesystemt::get();
  std::string root = register_tree("resolve", fs);

  auto f = fs.read(root + "/include/a.h");
  REQUIRE(f.has_value());
  REQUIRE(f->is_bundled());
  REQUIRE(f->view() == "int a;");
  // Borrowed, not copied.
  REQUIRE(f->view().data() == hdr_bytes);

  REQUIRE(fs.exists(root + "/lib/sub/b.c"));
  REQUIRE(!fs.exists(root + "/nope.h"));
  REQUIRE(!fs.read(root + "/nope.h").has_value());
}

TEST_CASE("reads fall back to the real filesystem", "[core][util][filesystem]")
{
  auto &fs = file_operations::filesystemt::get();
  auto tmp = file_operations::create_tmp_file("esbmc-test-%%%%");
  fputs("on disk", tmp.file());
  fflush(tmp.file());

  auto f = fs.read(tmp.path());
  REQUIRE(f.has_value());
  REQUIRE(!f->is_bundled());
  REQUIRE(f->view() == "on disk");
  REQUIRE(fs.exists(tmp.path()));
}

TEST_CASE("list walks a prefix recursively", "[core][util][filesystem]")
{
  auto &fs = file_operations::filesystemt::get();
  std::string root = register_tree("listing", fs);

  auto all = fs.list(root);
  REQUIRE(all.size() == 2);
  REQUIRE(all[0] == root + "/include/a.h");
  REQUIRE(all[1] == root + "/lib/sub/b.c");

  // A prefix scan, so a narrower one selects a subtree.
  REQUIRE(fs.list(root + "/lib").size() == 1);
  REQUIRE(fs.list(root + "/absent").empty());
}

TEST_CASE("list matches whole path components", "[core][util][filesystem]")
{
  auto &fs = file_operations::filesystemt::get();
  std::string root = std::string(file_operations::ESBMC_VFS_ROOT) + "/boundary";
  fs.add_bundled(root + "/lib/a.c", src_bytes, sizeof(src_bytes) - 1);
  fs.add_bundled(root + "/libcxx/b.c", src_bytes, sizeof(src_bytes) - 1);
  // '-' sorts below '/', so this one is scanned before root + "/lib/a.c".
  fs.add_bundled(root + "/lib-old/c.c", src_bytes, sizeof(src_bytes) - 1);

  auto lib = fs.list(root + "/lib");
  REQUIRE(lib.size() == 1);
  REQUIRE(lib[0] == root + "/lib/a.c");
}

TEST_CASE("materialize reproduces the tree on disk", "[core][util][filesystem]")
{
  auto &fs = file_operations::filesystemt::get();
  std::string root = register_tree("materialize", fs);

  const std::string dir = fs.materialize(root, "esbmc-test-%%%%");

  // Nested parents are created on demand.
  REQUIRE(boost::filesystem::exists(dir + "/include/a.h"));
  REQUIRE(boost::filesystem::exists(dir + "/lib/sub/b.c"));
  REQUIRE(boost::filesystem::file_size(dir + "/lib/sub/b.c") == 6);

  auto written = fs.read(dir + "/include/a.h");
  REQUIRE(written.has_value());
  REQUIRE(!written->is_bundled());
  REQUIRE(written->view() == "int a;");
}

TEST_CASE("materialize writes a subtree once", "[core][util][filesystem]")
{
  auto &fs = file_operations::filesystemt::get();
  std::string root = register_tree("once", fs);

  const std::string first = fs.materialize(root, "esbmc-test-%%%%");
  boost::filesystem::remove(first + "/include/a.h");

  // Same directory back, and nothing rewritten: the removed file stays gone.
  const std::string second = fs.materialize(root, "esbmc-test-%%%%");
  REQUIRE(second == first);
  REQUIRE(!boost::filesystem::exists(first + "/include/a.h"));
}
