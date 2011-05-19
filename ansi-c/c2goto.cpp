#include <goto-programs/goto_convert_functions.h>
#include <langapi/language_ui.h>
#include <ui_message.h>
#include <parseoptions.h>
#include <util/config.h>

class c2goto_parseopt : public parseoptions_baset, public language_uit
{
  public:
  c2goto_parseopt(int argc, const char **argv):
    parseoptions_baset("(16)(32)(64)", argc, argv),
    language_uit(cmdline)
  {
  }

  int doit() {
    goto_functionst goto_functions;

    config.set(cmdline);

    if (parse()) return true;
    if (typecheck()) return true;
    if (final()) return true;

    clear_parse();

    goto_convert(context, optionst(), goto_functions, ui_message_handler);

    return 0;
  }
};

int main(int argc, const char **argv)
{
  c2goto_parseopt parseopt(argc, argv);
  return parseopt.main();
}
