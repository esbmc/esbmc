#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <esbmc/esbmc_parseoptions.h>
#include <python-frontend/python_language.h>
#include <langapi/mode.h>

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

namespace bfs = boost::filesystem;

const mode_table_et mode_table[] = {
  LANGAPI_MODE_CLANG_C,
  LANGAPI_MODE_PYTHON,
  LANGAPI_MODE_END};

const std::string *internal_libc_header_dir()
{
  return nullptr;
}

struct parse_options : esbmc_parseoptionst
{
  using esbmc_parseoptionst::esbmc_parseoptionst;
  using esbmc_parseoptionst::parse_goto_program;
};

struct TempFile
{
  bfs::path path;
  explicit TempFile()
  {
    path = bfs::temp_directory_path() / bfs::unique_path("esbmc-%%%%-%%%%.py");
    bfs::ofstream out(path);
    std::string_view contents = "print('ok')\n";
    out << contents;
  }
  ~TempFile()
  {
    boost::system::error_code ec;
    bfs::remove(path, ec);
  }
};

TEST_CASE(
  "Check force-malloc-success for Python",
  "[esbmc][parse options for Python]")
{
  TempFile main_py;
  const char *argv[] = {main_py.path.string().c_str()};

  parse_options parser(1, argv);
  parser.cmdline.args.push_back(argv[0]);

  optionst options;
  goto_functionst goto_functions;
  parser.parse_goto_program(options, goto_functions);

  REQUIRE(options.get_bool_option("force-malloc-success") == true);
}
