#include <fstream>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/write_goto_binary.h>
#include <langapi/language_ui.h>
#include <langapi/mode.h>
#include <util/cmdline.h>
#include <util/config.h>
#include <util/irep2.h>
#include <util/parseoptions.h>
#include <util/ui_message.h>

const struct opt_templ c2goto_options[] = {
{ 0,	"16",		switc,		"" },
{ 0,	"32",		switc,		"" },
{ 0,	"64",		switc,		"" },
{ 0,  "floatbv",   switc,    "" },
{ 0,	"output",	string,		"" },
{ 'I',	"",		string,		"" },
{ 'D',	"",		string,		"" },
{ 0,	"",		switc,		"" }
};

class c2goto_parseopt : public parseoptions_baset, public language_uit
{
  public:
  c2goto_parseopt(int argc, const char **argv):
    parseoptions_baset(c2goto_options, argc, argv),
    language_uit(cmdline)
  {
  }

  int doit() override {
    goto_functionst goto_functions;

    config.set(cmdline);
    config.options.set_option("keep-unused", true);

    if (!cmdline.isset("output")) {
      std::cerr << "Must set output file" << std::endl;
      return 1;
    }

    if (parse()) return 1;
    if (typecheck()) return 1;

    std::ofstream out(cmdline.getval("output"), std::ios::out | std::ios::binary);

    if (write_goto_binary(out, context, goto_functions)) {
      std::cerr << "Failed to write C library to binary obj" << std::endl;
      return 1;
    }

    return 0;
  }
};

int main(int argc, const char **argv)
{
  // To avoid the static initialization fiasco,
  type_poolt bees(true);
  type_pool = bees;

  c2goto_parseopt parseopt(argc, argv);
  return parseopt.main();
}

const mode_table_et mode_table[] =
{
#ifndef WITHOUT_CLANG
  LANGAPI_HAVE_MODE_CLANG_C,
#endif
  LANGAPI_HAVE_MODE_C,
#ifndef WITHOUT_CLANG
  LANGAPI_HAVE_MODE_CPP,
#endif
  LANGAPI_HAVE_MODE_CPP,
  LANGAPI_HAVE_MODE_END
};
