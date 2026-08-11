#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "goto_factory.h"
#include <util/base/filesystem.h>

#include <boost/filesystem.hpp>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
namespace
{
/* Restores TMPDIR from the destructor: a staging call that throws would
 * otherwise leave the rest of the binary pointing at the removed sandbox. */
class tmpdir_override
{
  const char *_saved = getenv("TMPDIR");
  const std::string _old = _saved ? _saved : "";

public:
  explicit tmpdir_override(const std::string &dir)
  {
    setenv("TMPDIR", dir.c_str(), 1);
  }

  ~tmpdir_override()
  {
    if (_saved)
      setenv("TMPDIR", _old.c_str(), 1);
    else
      unsetenv("TMPDIR");
  }
};

/* goto_factory resolves the temp directory per call, so TMPDIR confines the
 * staged directory to a private sandbox and keeps this off the shared /tmp,
 * where a concurrently running unit test would stage one of its own. */
template <typename F>
std::vector<std::string> stage_and_list_leftovers(F &&stage)
{
  namespace fs = boost::filesystem;
  auto sandbox = file_operations::create_tmp_dir("esbmc-test-sandbox-%%%%");
  {
    tmpdir_override tmpdir(sandbox.path());
    program p = stage();
    REQUIRE(
      p.functions.function_map.find("c:@F@main") !=
      p.functions.function_map.end());
  }

  // The frontends extract their headers into the same directory and remove
  // them at exit, so only what goto_factory staged is of interest here.
  std::vector<std::string> staged;
  for (const fs::directory_entry &e : fs::directory_iterator(sandbox.path()))
  {
    const std::string name = e.path().filename().string();
    if (name.rfind("esbmc-test-", 0) == 0)
      staged.push_back(name);
  }
  return staged;
}
} // namespace

TEST_CASE(
  "goto_factory removes the directory it staged a string source in",
  "[testing-utils]")
{
  std::vector<std::string> staged = stage_and_list_leftovers(
    []
    {
      std::string code = "int main() { return 0; }";
      return goto_factory::get_goto_functions(code);
    });
  CHECK(staged == std::vector<std::string>{});
}

TEST_CASE(
  "goto_factory removes the directory it staged a stream source in",
  "[testing-utils]")
{
  std::vector<std::string> staged = stage_and_list_leftovers(
    []
    {
      std::istringstream code("int main() { return 0; }");
      return goto_factory::get_goto_functions(code);
    });
  CHECK(staged == std::vector<std::string>{});
}
#endif
