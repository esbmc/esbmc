/*******************************************************************\

Module: Command Line Parsing

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_CBMC_PARSEOPTIONS_H
#define CPROVER_CBMC_PARSEOPTIONS_H

#include <langapi/language_ui.h>
#include <ui_message.h>
#include <parseoptions.h>

#include "bmc.h"

#define CBMC_OPTIONS \
  "(program-only)(function):(preprocess)(slice-by-trace):" \
  "(no-simplify)(unwind):(unwindset):(slice-formula)" \
  "(debug-level):(no-substitution)(no-simplify-if)" \
  "(no-bounds-check)(cvc)(z3-bv)(z3-ir)(boolector-bv)(z3)(bl)(smt)(outfile):(no-pointer-check)" \
  "(document-subgoals)(all-claims)D:I:(depth):" \
  "(no-div-by-zero-check)(no-unwinding-assertions)(no-assume-guarantee)" \
  "(partial-loops)(int-encoding)(ecp)(show-features)(memory-leak-check)" \
  "(no-pretty-names)(overflow-check)(beautify-greedy)(beautify-pbs)" \
  "(floatbv)(fixedbv)(no-assertions)(gui)(nan-check)" \
  "(dimacs)(minisat)(16)(32)(64)(little-endian)(big-endian)(refine)" \
  "(show-goto-functions)(show-value-sets)(xml-ui)(show-loops)" \
  "(show-symbol-table)(show-vcc)(show-claims)(claim):" \
  "(atomicity-check)(error-label):(verbosity):(binary):(no-library)" \
  "(version)(i386-linux)(i386-macos)(i386-win32)(ppc-macos)(unsigned-char)" \
  "(arrays-uf-always)(arrays-uf-never)(interpreter)(no-lock-check)(deadlock-check)" \
  "(string-abstraction)(no-arch)(eager)(lazy)(no-slice)(uw-model)(control-flow-test)" \
  "(round-to-nearest)(round-to-plus-inf)(round-to-minus-inf)(round-to-zero)" \
  "(qf_aufbv)(qf_auflira)(btor)" \
  "(context-switch):(no-por)(data-races-check)(DFS)(schedule)(all-runs)" \
  "(timeout):(memlimit):(state-hashing)(symex-trace)" \
  "(core-size):(smtlib-ileave-num):" \
  "(decide)" // legacy, and will eventually disappear

class cbmc_parseoptionst:
  public parseoptions_baset,
  public language_uit
{
public:
  virtual int doit();
  virtual void help();

  cbmc_parseoptionst(int argc, const char **argv):
    parseoptions_baset(CBMC_OPTIONS, argc, argv),
    language_uit(cmdline)
  {
  }

  cbmc_parseoptionst(
    int argc,
    const char **argv,
    const std::string &extra_options):
    parseoptions_baset(CBMC_OPTIONS+extra_options, argc, argv),
    language_uit(cmdline)
  {
  }

protected:
  virtual void get_command_line_options(optionst &options);
  virtual int do_bmc(bmc_baset &bmc, const goto_functionst &goto_functions);

  virtual bool get_goto_program(
    bmc_baset &bmc,
    goto_functionst &goto_functions);

  virtual bool process_goto_program(
    bmc_baset &bmc,
    goto_functionst &goto_functions);

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
  exprt calculate_a_property_monitor(std::string prefix, std::map<std::string, std::string> &strings, std::set<std::string> &used_syms);
  void add_monitor_exprs(goto_programt::targett insn, goto_programt::instructionst &insn_list, std::map<std::string, std::pair<std::set<std::string>, exprt> >monitors);
};

#endif
