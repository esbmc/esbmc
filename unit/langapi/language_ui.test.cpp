/*******************************************************************\
Module: Unit tests for language_uit::parse()
\*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>

#include "../testing-utils/goto_factory.h"
#include <clang-c-frontend/AST/vfs_paths.h>
#include <langapi/language_ui.h>
#include <util/base/filesystem.h>
#include <util/config/config.h>

#include <fstream>
#ifndef _WIN32
#  include <sys/stat.h>
#endif

/* Shaped like an operational model: c2goto names the sources it compiles by
 * their VFS path, and those exist only in .rodata. */
static const char probe_c[] = "int esbmc_vfs_probe(int x) { return x + 1; }\n";

static std::string bundle(const std::string &name)
{
  std::string path = clang_vfs_root() + "/unit/langapi/" + name;
  file_operations::filesystemt::get().add_bundled(
    path, probe_c, sizeof(probe_c) - 1);
  return path;
}

static void configure()
{
  cmdlinet cmd = goto_factory::get_default_cmdline("");
  config.set(cmd);
  config.ansi_c.set_data_model(configt::LP64);
  config.options = goto_factory::get_default_options(cmd);
}

TEST_CASE(
  "parse compiles a bundled source that is absent from disk",
  "[core][langapi]")
{
  configure();
  std::string path = bundle("probe.c");
  REQUIRE(!std::ifstream(path));

  language_uit l;
  REQUIRE(!l.parse(path));
  REQUIRE(!l.typecheck());

  bool found = false;
  l.context.foreach_operand([&found](const symbolt &s) {
    found |= s.name.as_string().find("esbmc_vfs_probe") != std::string::npos;
  });
  REQUIRE(found);
}

TEST_CASE("parse rejects a VFS path nothing registered", "[core][langapi]")
{
  configure();
  language_uit l;
  REQUIRE(l.parse(clang_vfs_root() + "/unit/langapi/absent.c"));
}

#ifndef _WIN32
TEST_CASE(
  "parse rejects a file that exists but does not open",
  "[core][langapi]")
{
  configure();
  auto tmp = file_operations::create_tmp_file("esbmc-test-%%%%%%.c");
  REQUIRE(chmod(tmp.path().c_str(), 0) == 0);

  if (std::ifstream(tmp.path()))
  {
    SUCCEED("mode 000 is no obstacle to this process, so nothing to reject");
    return;
  }

  REQUIRE(file_operations::filesystemt::get().exists(tmp.path()));
  language_uit l;
  REQUIRE(l.parse(tmp.path()));
}
#endif
