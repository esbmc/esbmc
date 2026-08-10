#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "goto_factory.h"
#include <util/base/filesystem.h>

#include <boost/filesystem.hpp>
#include <cstdlib>

#ifndef _WIN32
TEST_CASE(
  "goto_factory removes the directory it staged the source in",
  "[testing-utils]")
{
  namespace fs = boost::filesystem;
  auto sandbox = file_operations::create_tmp_dir("esbmc-test-sandbox-%%%%");

  // goto_factory resolves the temp directory per call, so TMPDIR confines the
  // staged directory to `sandbox` and keeps this off the shared /tmp, where a
  // concurrently running unit test would stage one of its own.
  const char *saved = getenv("TMPDIR");
  const std::string old_tmpdir = saved ? saved : "";
  setenv("TMPDIR", sandbox.path().c_str(), 1);

  std::string code = "int main() { return 0; }";
  program p = goto_factory::get_goto_functions(code);

  if (saved)
    setenv("TMPDIR", old_tmpdir.c_str(), 1);
  else
    unsetenv("TMPDIR");

  REQUIRE(
    p.functions.function_map.find("c:@F@main") !=
    p.functions.function_map.end());

  // The frontends extract their headers into the same directory and remove
  // them at exit, so only what goto_factory staged is of interest here.
  std::vector<std::string> staged;
  for (const fs::directory_entry &e : fs::directory_iterator(sandbox.path()))
  {
    const std::string name = e.path().filename().string();
    if (name.rfind("esbmc-test-", 0) == 0)
      staged.push_back(name);
  }
  CHECK(staged == std::vector<std::string>{});
}
#endif
