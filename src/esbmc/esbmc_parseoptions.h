#ifndef CPROVER_ESBMC_PARSEOPTIONS_H
#define CPROVER_ESBMC_PARSEOPTIONS_H

#include <esbmc/bmc.h>
#include <esbmc/kind_invariants.h>
#include <goto-programs/goto_convert_functions.h>
#include <langapi/language_ui.h>
#include <util/config/cmdline.h>
#include <util/config/options.h>
#include <util/config/parseoptions.h>
#include <util/ssa/algorithms.h>
#include <util/base/threeval.h>
#include <util/base/time_stopping.h>
#include <string_view>

// Macro to determine if color output should be enabled
#ifdef _WIN32
#  include <io.h>
#  define ENABLE_COLOR(val)                                                    \
    ((val) == "always" || ((val) == "auto" && _isatty(_fileno(stderr))))
#else
#  include <unistd.h>
#  define ENABLE_COLOR(val)                                                    \
    ((val) == "always" || ((val) == "auto" && isatty(fileno(stderr))))
#endif

extern const struct group_opt_templ all_cmd_options[];

class esbmc_parseoptionst : public parseoptions_baset, public language_uit
{
public:
  int doit() override;
  void help() override;
  const char *fatal_signal_advice() const override
  {
    return "Re-run with --segfault-handler for a backtrace.\n";
  }

  esbmc_parseoptionst(int argc, const char **argv)
    : parseoptions_baset(all_cmd_options, argc, argv)
  {
  }

  ~esbmc_parseoptionst()
  {
    close_file(out);
  }

protected:
  virtual void get_command_line_options(optionst &options);
  virtual int do_bmc(bmct &bmc);

  virtual bool
  get_goto_program(optionst &options, goto_functionst &goto_functions);

  virtual bool
  create_goto_program(optionst &options, goto_functionst &goto_functions);

  virtual bool
  parse_goto_program(optionst &options, goto_functionst &goto_functions);

  virtual bool
  process_goto_program(optionst &options, goto_functionst &goto_functions);

  /// Whether the run needs the loop-invariant machinery.
  bool wants_loop_invariants() const;

  /// Synthesise the invariants when asked, then run the schema over them.
  void apply_loop_invariants(
    goto_functionst &goto_functions,
    contextt &context,
    const optionst &options,
    bool k_induction_ran);

  virtual bool
  output_goto_program(optionst &options, goto_functionst &goto_functions);

  /// \brief Process function contracts if enabled
  /// \param goto_functions GOTO functions
  /// \param has_replace Whether to replace calls with contracts
  /// \param has_enforce Whether to enforce contracts
  /// \param has_enforce_all Whether to enforce contracts for all annotated functions
  /// \param has_replace_all Whether to replace calls for all annotated functions
  /// \return True on a usage error, e.g. a named function that nothing acted on
  bool process_function_contracts(
    goto_functionst &goto_functions,
    bool has_replace,
    bool has_enforce,
    bool has_enforce_all,
    bool has_replace_all);

  int do_bmc_strategy(optionst &options, goto_functionst &goto_functions);

  /// k-induction that raises each loop's bound to what the forward
  /// condition's counterexample shows it needs; see adaptive_kind_strategy.cpp.
  int do_adaptive_kind_strategy(
    optionst &options,
    goto_functionst &goto_functions);

  /// Houdini fixpoint over guessed loop-invariant candidates; see
  /// goto_houdini_invariants.h.
  int do_houdini_strategy(optionst &options, goto_functionst &goto_functions);

  /// Houdini fixpoint over the candidates of @p pool named by @p ids, on
  /// copies of @p pristine, which must not carry the k-induction transform.
  /// Each round solves the loop-invariant schema under @p probe_options, a
  /// base-case run, with each loop the schema leaves to the
  /// unwinder bounded as @p bounds says for its stamp. No round starts after
  /// @p deadline, as given by current_time(); a fixpoint not reached by then
  /// proves nothing. The global verdict table is left as it was found.
  kind_proof_resultt prove_kind_candidates(
    const goto_functionst &pristine,
    const std::vector<kind_candidatet> &pool,
    const std::vector<size_t> &ids,
    const optionst &probe_options,
    const std::map<unsigned, BigInt> &bounds,
    fine_timet deadline);

  int do_context_bound_deepening(
    optionst &options,
    goto_functionst &goto_functions);

  int falsify_with_bounded_schedules(
    optionst &options,
    goto_functionst &goto_functions);

  int run_chosen_strategy(optionst &options, goto_functionst &goto_functions);

  int doit_k_induction_parallel();

  tvt is_base_case_violated(
    optionst &options,
    goto_functionst &goto_functions,
    const uint64_t &k_step);

  /// \param hints when set, filled from a satisfiable result; see
  ///   bmct::infer_loop_bounds.
  tvt does_forward_condition_hold(
    optionst &options,
    goto_functionst &goto_functions,
    const uint64_t &k_step,
    bmct::kind_feedbackt *hints = nullptr);

  /// \param feedback when set, filled from a satisfiable result with the
  ///   loop-head states the counterexample passes through.
  tvt is_inductive_step_violated(
    optionst &options,
    goto_functionst &goto_functions,
    const uint64_t &k_step,
    bmct::kind_feedbackt *feedback = nullptr);

  void diagnose_unknown_properties(
    optionst &options,
    goto_functionst &goto_functions,
    uint64_t k_step);

  bool read_goto_binary(goto_functionst &goto_functions);

  /// True if any --binary input is a CBMC goto-binary (magic 0x7f 'G' 'B' 'F').
  bool has_cbmc_binary_input();

  /// Synthesises ESBMC's "additions" (the __ESBMC_main entry wrapper and the
  /// CPROVER-intrinsic bodies) by compiling a boilerplate translation unit
  /// through the C frontend into the given symbol table / goto functions, so a
  /// CBMC goto-binary verifies without manually linking a library.goto.
  /// Returns true on error.
  bool synthesize_cprover_additions(
    optionst &options,
    goto_functionst &goto_functions);

  bool set_claims(goto_functionst &goto_functions);

  uint64_t read_time_spec(std::string_view str);
  uint64_t read_mem_spec(std::string_view str);

  void preprocessing();

  void add_property_monitors(goto_functionst &goto_functions, namespacet &ns);
  expr2tc calculate_a_property_monitor(
    const std::string &prefix,
    std::set<std::string> &used_syms) const;
  void add_monitor_exprs(
    goto_programt::targett insn,
    goto_programt::instructionst &insn_list,
    const std::map<std::string, std::pair<std::set<std::string>, expr2tc>>
      &monitors);

  void print_ileave_points(namespacet &ns, goto_functionst &goto_functions);

  FILE *out = stderr;

  std::vector<std::unique_ptr<goto_functions_algorithm>>
    goto_preprocess_algorithms;

  // Dead-store advisories (CWE-563) collected by --dead-store-check during
  // goto preprocessing; surfaced textually and (via bmct) in SARIF.
  std::vector<dead_store_advisoryt> dead_store_advisories;

  // coverage mode
  bool is_coverage;

private:
  bool resolve_color_option() const;
  void close_file(FILE *f)
  {
    if (f != stdout && f != stderr)
    {
      fclose(f);
    }
  }

public:
  goto_functionst goto_functions;
};

#endif
