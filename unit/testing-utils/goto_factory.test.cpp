#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "goto_factory.h"
#include <util/base/filesystem.h>

#include <boost/filesystem.hpp>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace
{
/* boost::filesystem::temp_directory_path() consults TMPDIR first on POSIX and
 * TMP first on Windows, so this is the variable that moves the staged path. */
#ifdef _WIN32
constexpr const char *tmpdir_var = "TMP";
#else
constexpr const char *tmpdir_var = "TMPDIR";
#endif

/* A null value removes the variable. */
void set_tmpdir(const char *value)
{
#ifdef _WIN32
  _putenv_s(tmpdir_var, value ? value : "");
#else
  if (value)
    setenv(tmpdir_var, value, 1);
  else
    unsetenv(tmpdir_var);
#endif
}

/* Restores the temp directory from the destructor: a staging call that throws
 * would otherwise leave the rest of the binary pointing at the removed
 * sandbox. */
class tmpdir_override
{
  const bool _was_set = getenv(tmpdir_var) != nullptr;
  const std::string _old = _was_set ? getenv(tmpdir_var) : "";

public:
  explicit tmpdir_override(const std::string &dir)
  {
    set_tmpdir(dir.c_str());
  }

  ~tmpdir_override()
  {
    set_tmpdir(_was_set ? _old.c_str() : nullptr);
  }
};

/* goto_factory resolves the temp directory per call, so the override confines
 * the staged directory to a private sandbox and keeps this off the shared
 * temp directory, where a concurrent unit test would stage one of its own. */
template <typename F>
std::vector<std::string> stage_and_list_leftovers(F &&stage)
{
  namespace fs = boost::filesystem;
  auto sandbox = file_operations::create_tmp_dir("esbmc-test-sandbox-%%%%");
  {
    tmpdir_override tmpdir(sandbox.path());

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
  std::vector<std::string> staged = stage_and_list_leftovers([] {
    std::string code = "int main() { return 0; }";
    return goto_factory::get_goto_functions(code);
  });
  CHECK(staged == std::vector<std::string>{});
}

TEST_CASE(
  "goto_factory removes the directory it staged a stream source in",
  "[testing-utils]")
{
  std::vector<std::string> staged = stage_and_list_leftovers([] {
    std::istringstream code("int main() { return 0; }");
    return goto_factory::get_goto_functions(code);
  });
  CHECK(staged == std::vector<std::string>{});
}
