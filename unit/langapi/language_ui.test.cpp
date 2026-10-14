/*******************************************************************\
Module: Unit tests for language_uit::parse()
\*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this
                          // in one cpp file
#include <catch2/catch.hpp>

#include "../testing-utils/goto_factory.h"
#include <clang-c-frontend/AST/vfs_paths.h>
#include <langapi/language_ui.h>
#include <util/base/filesystem.h>

#include <fstream>
#ifndef _WIN32
#  include <cstring>
#  include <sys/socket.h>
#  include <sys/un.h>
#  include <unistd.h>
#endif

static const char probe_c[] = "int esbmc_vfs_probe(int x) { return x + 1; }\n";

TEST_CASE(
  "parse compiles a bundled source that is absent from disk",
  "[core][langapi]")
{
  cmdlinet cmd = goto_factory::get_default_cmdline("");
  optionst opts = goto_factory::get_default_options(cmd);
  goto_factory::config_environment(
    goto_factory::Architecture::BIT_64, cmd, opts);

  std::string path = clang_vfs_root() + "/unit/langapi/probe.c";
  file_operations::filesystemt::get().add_bundled(
    path, probe_c, sizeof(probe_c) - 1);
  REQUIRE(!std::ifstream(path));

  language_uit l;
  REQUIRE(!l.parse(path));
  REQUIRE(!l.typecheck());
  REQUIRE(l.context.find_symbol("c:@F@esbmc_vfs_probe"));
}

TEST_CASE("parse rejects a VFS path nothing registered", "[core][langapi]")
{
  language_uit l;
  REQUIRE(l.parse(clang_vfs_root() + "/unit/langapi/absent.c"));
}

#ifndef _WIN32
TEST_CASE(
  "parse rejects a file that exists but does not open",
  "[core][langapi]")
{
  // A socket is the one file open(2) refuses whatever the uid, so unlike mode
  // 000 this still asserts something when the suite runs as root.
  auto dir = file_operations::create_tmp_dir("esbmc-test-%%%%%%");
  std::string path = dir.path() + "/probe.c";

  sockaddr_un addr = {};
  addr.sun_family = AF_UNIX;
  REQUIRE(path.size() < sizeof(addr.sun_path));
  memcpy(addr.sun_path, path.c_str(), path.size());

  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);
  int bound = bind(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr));
  close(fd);
  REQUIRE(bound == 0);

  REQUIRE(file_operations::filesystemt::get().exists(path));
  language_uit l;
  REQUIRE(l.parse(path));
}
#endif
