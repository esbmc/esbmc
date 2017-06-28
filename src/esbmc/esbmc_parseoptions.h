/*******************************************************************\

Module: Command Line Parsing

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_CBMC_PARSEOPTIONS_H
#define CPROVER_CBMC_PARSEOPTIONS_H

#include <esbmc/bmc.h>
#include <goto-programs/goto_convert_functions.h>
#include <langapi/language_ui.h>
#include <util/cmdline.h>
#include <util/options.h>
#include <util/parseoptions.h>
#include <util/ui_message.h>

extern const struct opt_templ esbmc_options[];

class cbmc_parseoptionst:
  public parseoptions_baset,
  public language_uit
{
public:
  int doit() override ;
  void help() override ;

  cbmc_parseoptionst(int argc, const char **argv):
    parseoptions_baset(esbmc_options, argc, argv),
    language_uit(cmdline)
  {
  }

protected:
  virtual void get_command_line_options(optionst &options);
  virtual int do_bmc(bmct &bmc);

  virtual bool get_goto_program(
    optionst &options,
    goto_functionst &goto_functions);

  virtual bool process_goto_program(
    optionst &options,
    goto_functionst &goto_functions);

  int doit_k_induction();
  int doit_k_induction_parallel();

  int doit_falsification();
  int doit_incremental();

  int do_base_case(optionst &opts, goto_functionst &goto_functions, int k_step);
  int do_forward_condition(optionst &opts, goto_functionst &goto_functions, int k_step);
  int do_inductive_step(optionst &opts, goto_functionst &goto_functions, int k_step);

  bool read_goto_binary(goto_functionst &goto_functions);

  bool set_claims(goto_functionst &goto_functions);

  void set_verbosity_msg(messaget &message);

  uint64_t read_time_spec(const char *str);
  uint64_t read_mem_spec(const char *str);

  void preprocessing();

  void add_property_monitors(goto_functionst &goto_functions, namespacet &ns);
  expr2tc calculate_a_property_monitor(const std::string&& prefix, std::map<std::string, std::string> &strings, std::set<std::string> &used_syms);
  void add_monitor_exprs(goto_programt::targett insn, goto_programt::instructionst &insn_list, std::map<std::string, std::pair<std::set<std::string>, expr2tc> >monitors);

  void print_ileave_points(namespacet &ns, goto_functionst &goto_functions);

public:
  goto_functionst goto_functions;
};

#endif
