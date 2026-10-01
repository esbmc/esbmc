#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "goto_factory.h"
#include <util/base/filesystem.h>

#include <boost/filesystem.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace
{
/* goto_factory resolves the temp directory per call, so the override confines
 * the staged directory to a private sandbox and keeps this off the shared
 * temp directory, where a concurrent unit test would stage one of its own. */
template <typename F>
std::vector<std::string> stage_and_list_leftovers(F &&stage)
{
  namespace fs = boost::filesystem;
  auto sandbox = file_operations::create_tmp_dir("esbmc-test-sandbox-%%%%");
  {
    file_operations::tmp_dir_override tmpdir(sandbox.path());

    /* A staging path that stopped honouring the override would leave the
     * sandbox empty and the check below green, so pin that a directory named
     * the way goto_factory names its own lands here. */
    file_operations::tmp_path probe(
      file_operations::get_unique_tmp_path("esbmc-test-%%%%%%"));
    REQUIRE(
      fs::equivalent(fs::path(probe.path()).parent_path(), sandbox.path()));

    program p = stage();
    REQUIRE(
      p.functions.function_map.find("c:@F@main") !=
      p.functions.function_map.end());
  }

  std::vector<std::string> leftovers;
  for (const fs::directory_entry &e : fs::directory_iterator(sandbox.path()))
    leftovers.push_back(e.path().filename().string());
  return leftovers;
}
} // namespace

TEST_CASE(
  "goto_factory removes the directory it staged a string source in",
  "[testing-utils]")
{
  std::vector<std::string> leftovers = stage_and_list_leftovers([] {
    std::string code = "int main() { return 0; }";
    return goto_factory::get_goto_functions(code);
  });
  CHECK(leftovers == std::vector<std::string>{});
}

TEST_CASE(
  "goto_factory removes the directory it staged a stream source in",
  "[testing-utils]")
{
  std::vector<std::string> leftovers = stage_and_list_leftovers([] {
    std::istringstream code("int main() { return 0; }");
    return goto_factory::get_goto_functions(code);
  });
  CHECK(leftovers == std::vector<std::string>{});
}
