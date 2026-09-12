#include <cstdlib>
#include <fstream>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/write_goto_binary.h>
#include <langapi/language_ui.h>
#include <langapi/mode.h>
#include <python-frontend/python_language.h>
#include <util/config/cmdline.h>
#include <util/config/config.h>
#include <irep2/irep2.h>
#include <util/config/parseoptions.h>

const struct group_opt_templ python2goto_options[] = {
  {"Basic Usage",
   {{"input-file",
     boost::program_options::value<std::vector<std::string>>()->value_name(
       "file.py"),
     "entry module compiled for its operational models"}}},
  {"Options",
   {
     {"output",
      boost::program_options::value<std::string>()->value_name("<filename>"),
      "write the goto binary to this file"},
     {"python",
      boost::program_options::value<std::string>()->value_name("path"),
      "Python interpreter binary to use (searched in $PATH; default: python)"},
     {"verbosity",
      boost::program_options::value<std::vector<std::string>>(),
      "Verbosity of log output, can be given multiple times. Parameter is "
      "either a decimal N or 'module:N' to set the log-level of debug messages "
      "of the module to N; without module, it sets the global log-level"},
   }},
  {"end", {{"", NULL, "end of options"}}},
  {"Hidden Options", {{"", NULL, ""}}}};

class python2goto_parseopt : public parseoptions_baset, public language_uit
{
public:
  python2goto_parseopt(int argc, const char **argv)
    : parseoptions_baset(python2goto_options, argc, argv)
  {
  }

  int doit() override
  {
    goto_functionst goto_functions;

    if (config.set(cmdline))
      return 1;
    config.options.cmdline(cmdline);
    // Suppresses the per-program parts of python_converter::convert(): the
    // C library, __name__/__file__, and the python_init/python_user_main/main
    // trio. What is left is the operational models' symbols.
    config.options.set_option("building-python-library", true);
    set_verbosity_msg(VerbosityLevel::Result);

    if (!cmdline.isset("output"))
    {
      log_error("Must set output file");
      return 1;
    }

    /* The models are named by their VFS paths, so they have to be in the
     * registry before the forked parser tries to open one. */
    python_languaget::register_bundled();

    if (parse(cmdline))
      return 1;
    if (typecheck())
      return 1;

    std::ofstream out(
      cmdline.getval("output"), std::ios::out | std::ios::binary);

    if (write_goto_binary(out, context, goto_functions))
    {
      log_error("Failed to write Python models to binary obj");
      return 1;
    }

    return 0;
  }
};

int main(int argc, const char **argv)
{
  // Disable config loading because it interferes with PYTHON2GOTO program.
  char env[] = "ESBMC_CONFIG_FILE=";
  putenv(env);
  python2goto_parseopt parseopt(argc, argv);
  return parseopt.main();
}

const mode_table_et mode_table[] = {
  LANGAPI_MODE_CLANG_C,
  LANGAPI_MODE_CLANG_CPP,
  LANGAPI_MODE_PYTHON,
  LANGAPI_MODE_END};
