/*******************************************************************\

Module: Command Line Parsing

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_CBMC_PARSEOPTIONS_H
#define CPROVER_CBMC_PARSEOPTIONS_H

#include <langapi/language_ui.h>
#include <ui_message.h>
#include <parseoptions.h>

#include <options.h>
#include <cmdline.h>

#include "bmc.h"

extern const struct opt_templ esbmc_options[];

class cbmc_parseoptionst:
  public parseoptions_baset,
  public language_uit
{
public:
  virtual int doit();
  virtual void help();

  cbmc_parseoptionst(int argc, const char **argv):
    parseoptions_baset(esbmc_options, argc, argv),
    language_uit(cmdline)
  {
  }

protected:
  optionst options;

  virtual void get_command_line_options(optionst &options);
  virtual int do_bmc(bmct &bmc, const goto_functionst &goto_functions);

  virtual bool get_goto_program(goto_functionst &goto_functions);

  virtual bool process_goto_program(goto_functionst &goto_functions);

  bool read_goto_binary(goto_functionst &goto_functions);

  bool set_claims(goto_functionst &goto_functions);

  void set_verbosity(messaget &message);

  // get any additional stuff before finalizing
  virtual bool get_modules()
  {
    return false;
  }

  void preprocessing();

  void add_property_monitors(goto_functionst &goto_functions);
  expr2tc calculate_a_property_monitor(std::string prefix, std::map<std::string, std::string> &strings, std::set<std::string> &used_syms);
  void add_monitor_exprs(goto_programt::targett insn, goto_programt::instructionst &insn_list, std::map<std::string, std::pair<std::set<std::string>, expr2tc> >monitors);

  void print_ileave_points(namespacet &ns, goto_functionst &goto_functions);
};

#endif
