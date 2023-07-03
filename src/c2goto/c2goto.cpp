#include <fstream>
#include <memory>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/write_goto_binary.h>
#include <langapi/language_ui.h>
#include <langapi/mode.h>
#include <util/cmdline.h>
#include <util/config.h>
#include <irep2/irep2.h>
#include <util/parseoptions.h>

const struct group_opt_templ c2goto_options[] = {
  {"Basic Usage",
   {{"input-file",
     boost::program_options::value<std::vector<std::string>>()->value_name(
       "file.c ..."),
     "source file names"}}},
  {"Options",
   {{"16", NULL, "set width of machine word (default is 64)"},
    {"32", NULL, "set width of machine word (default is 64)"},
    {"64", NULL, "set width of machine word (default is 64)"},
    {"fixedbv", NULL, "encode floating-point as fixed bit-vectors"},
    {"floatbv",
     NULL,
     "encode floating-point using the SMT floating-point theory (default)"},
    {"output",
     boost::program_options::value<std::string>()->value_name("<filename>"),
     "output VCCs in SMT lib format to given file"},
    {"include,I",
     boost::program_options::value<std::vector<std::string>>()->value_name(
       "path"),
     "set include path"},
    {"idirafter",
     boost::program_options::value<std::vector<std::string>>()->value_name(
       "path"),
     "append system include path to search after system headers"},
    {"define,D",
     boost::program_options::value<std::vector<std::string>>()->value_name(
       "macro"),
     "define preprocessor macro"},
    {"sysroot",
     boost::program_options::value<std::string>()->value_name("<path>"),
     "set the sysroot for the frontend"}

   }},
  {"end", {{"", NULL, "end of options"}}},
  {"Hidden Options", {{"", NULL, ""}}}};

class c2goto_parseopt : public parseoptions_baset, public language_uit
{
public:
  c2goto_parseopt(int argc, const char **argv)
    : parseoptions_baset(c2goto_options, argc, argv), language_uit(cmdline)
  {
  }

  int doit() override
  {
    goto_functionst goto_functions;

    if(config.set(cmdline))
      return 1;
    config.options.cmdline(cmdline);
    messaget::state.verbosity = VerbosityLevel::Result;

    if(!cmdline.isset("output"))
    {
      log_error("Must set output file");
      return 1;
    }

    if(parse())
      return 1;
    if(typecheck())
      return 1;

    std::ofstream out(
      cmdline.getval("output"), std::ios::out | std::ios::binary);

    if(write_goto_binary(out, context, goto_functions))
    {
      log_error("Failed to write C library to binary obj");
      return 1;
    }

    return 0;
  }
};

int main(int argc, const char **argv)
{
  c2goto_parseopt parseopt(argc, argv);
  return parseopt.main();
}

const mode_table_et mode_table[] = {LANGAPI_MODE_CLANG_C, LANGAPI_MODE_END};
