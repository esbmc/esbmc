#include <ac_config.h>

#ifndef _WIN32
extern "C"
{
#  include <fcntl.h>
#  include <unistd.h>

#  ifdef HAVE_SENDFILE_ESBMC
#    include <sys/sendfile.h>
#  endif

#  include <sys/resource.h>
#  include <sys/time.h>
#  include <sys/types.h>
}
#endif

#include <esbmc/bmc.h>
#include <esbmc/esbmc_parseoptions.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/sarif.h>
#include <util/cwe_mapping.h>
#include <solvers/smt/smt_result.h>
#include <solvers/smtlib/smtlib_conv.h>
#include <solvers/solve.h>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <clang-c-frontend/clang_c_language.h>
#include <util/config.h>
#include <util/filesystem.h>
#include <csignal>
#include <cstdlib>
#include <limits>
#include <util/expr_util.h>
#include <iostream>
#include <fstream>
#include <goto-programs/add_race_assertions.h>
#include <goto-programs/add_restrict_assertions.h>
#include <goto-programs/goto_atomicity_check.h>
#include <goto-programs/goto_check.h>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_inline.h>
#include <goto-programs/goto_k_induction.h>
#include <goto-programs/goto_termination.h>
#include <esbmc/ranking_synthesis.h>
#include <esbmc/non_termination.h>
#include <goto-programs/goto_loop_simplify.h>
#include <goto-programs/goto_loop_invariant.h>
#include <goto-programs/abstract-interpretation/interval_analysis.h>
#include <goto-programs/abstract-interpretation/gcse.h>
#include <goto-programs/loop_numbers.h>
#include <goto-programs/goto_binary_reader.h>
#include <goto-programs/read_cbmc_goto_object.h>
#include <goto-programs/write_goto_binary.h>
#include <goto-programs/remove_no_op.h>
#include <goto-programs/remove_unreachable.h>
#include <goto-programs/remove_exceptions.h>
#include <goto-programs/set_claims.h>
#include <goto-programs/show_claims.h>
#include <goto-programs/loop_unroll.h>
#include <goto-programs/goto_check_uninit_vars.h>
#include <goto-programs/goto_check_unchecked_return.h>
#include <goto-programs/goto_check_excessive_alloc.h>
#include <goto-programs/dead_store_analysis.h>
#include <goto-programs/mark_decl_as_non_det.h>
#include <goto-programs/assign_params_as_non_det.h>
#include <goto2c/goto2c.h>
#include <util/irep.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <memory>
#include <pointer-analysis/goto_program_dereference.h>
#include <pointer-analysis/show_value_sets.h>
#include <pointer-analysis/value_set_analysis.h>
#include <util/symbol.h>
#include <util/time_stopping.h>
#include <goto-programs/goto_cfg.h>
#include <langapi/language_util.h>
#include <goto-programs/contracts/contracts.h>

#ifndef _WIN32
#  include <sys/wait.h>
#  include <fcntl.h>
#  ifdef __GLIBC__
#    include <execinfo.h>
#  endif
#endif

#ifdef ENABLE_GOTO_CONTRACTOR
#  include <goto-programs/goto_contractor.h>
#endif

// ANSI color/style escape sequences for terminal output
#define CLR_BOLD_CYAN "\033[1;36m"
#define CLR_BOLD "\033[1m"
#define CLR_RESET "\033[0m"

// This is the main entry point of ESBMC. Here ESBMC performs initialisation
// of the algorithms that will be run over the GOTO program at later stages
//
//  1) Parse CMD                            (see "get_command_line_options")
//  2) Create and preprocess a GOTO program (see "get_goto_functions")
//  3) Set user-specified claims            (see "set_claims")
//  4) Perform Bounded Model Checking
//    - Run a particular verification strategy if specified
//      in CMD (see "do_bmc_strategy"), or
//    - Perform a single run of Bounded Model Checking and rely
//      on the simplifier to determine the sufficient verification bound
//      (see "do_bmc")
int esbmc_parseoptionst::doit()
{
  // Configure msg output
  if (cmdline.isset("file-output"))
  {
    FILE *f = fopen(cmdline.getval("file-output"), "w+");
    /* TODO: handle failure */
    out = f;
    messaget::state.out = f;
  }

  // Print a banner with version info to stdout
  {
    FILE *output_stream = messaget::state.out;
    messaget::state.out = stdout;
    log_status(
      "ESBMC version {} {}-bit {} {}",
      ESBMC_VERSION,
      sizeof(void *) * 8,
      config.this_architecture(),
      config.this_operating_system());
    messaget::state.out = output_stream;
  }

  if (cmdline.isset("version"))
    return 0;

  // Unwinding of transition systems
  if (cmdline.isset("module") || cmdline.isset("gen-interface"))
  {
    log_error("This version has no support for hardware modules.");
    return 1;
  }

  // Preprocess the input program.
  // (This will not have any effect if OLD_FRONTEND is not enabled.)
  if (cmdline.isset("preprocess"))
  {
    preprocessing();
    return 0;
  }

  // Initialize goto_functions algorithms
  {
    // Loop unrolling
    if (cmdline.isset("goto-unwind") && !cmdline.isset("unwind"))
    {
      size_t unroll_limit = cmdline.isset("unlimited-goto-unwind") ? -1 : 1000;
      goto_preprocess_algorithms.push_back(
        std::make_unique<bounded_loop_unroller>(unroll_limit));
    }

    // Unroll intrinsic support
    goto_preprocess_algorithms.emplace_back(
      std::make_unique<apply_intrinsic_unroller>());

    // Uninitialised-variable check (CWE-457) must run before
    // mark_decl_as_non_det, which would otherwise overwrite every
    // uninitialised DECL with a nondet ASSIGN and erase the property.
    if (cmdline.isset("uninitialised-vars-check"))
      goto_preprocess_algorithms.emplace_back(
        std::make_unique<goto_check_uninit_vars>(context));

    // Unchecked-return-value check (CWE-252). Runs as a preprocessing
    // algorithm so the inserted ASSERTs participate in the same path-
    // condition pruning as the rest of the goto-program.
    if (cmdline.isset("unchecked-return-value-check"))
      goto_preprocess_algorithms.emplace_back(
        std::make_unique<goto_check_unchecked_return>(context));

    // Excessive-allocation-size check (CWE-789). The bound K is the byte
    // limit above which an allocation size is flagged; a bare flag uses the
    // implicit 1 MiB default (see options.cpp).
    if (cmdline.isset("excessive-alloc-check"))
    {
      int k = atoi(cmdline.getval("excessive-alloc-check"));
      if (k <= 0)
      {
        log_error("--excessive-alloc-check=K requires K >= 1 (got {})", k);
        return 1;
      }
      goto_preprocess_algorithms.emplace_back(
        std::make_unique<goto_check_excessive_alloc>(context, BigInt(k)));
    }

    // Dead-store advisory (CWE-563). Must also run before mark_decl_as_non_det,
    // which rewrites uninitialised DECLs into `DECL; ASSIGN x = nondet` — that
    // synthetic store would otherwise be reported as a spurious dead store.
    if (cmdline.isset("dead-store-check"))
      goto_preprocess_algorithms.emplace_back(
        std::make_unique<goto_check_dead_store>(
          context, dead_store_advisories));

    // Explicitly marking all declared variables as "nondet"
    goto_preprocess_algorithms.emplace_back(
      std::make_unique<mark_decl_as_non_det>(context));

    if (cmdline.isset("function") && cmdline.isset("assign-param-nondet"))
    {
      // assign parameters to "nondet"
      goto_preprocess_algorithms.emplace_back(
        std::make_unique<assign_params_as_non_det>(context));
    }
  }

  // Run this before the main flow. This method performs its own
  // parsing and preprocessing.
  // This is an old implementation of parallel k-induction algorithm.
  // Eventually we will modify it and implement parallel version for all
  // available strategies. Just run it first before everything else
  // for now.
  //
  // The parallel driver has no termination interpretation, so --termination
  // takes priority: fall through to the sequential strategy instead (#6031).
  if (cmdline.isset("k-induction-parallel") && !cmdline.isset("termination"))
    return doit_k_induction_parallel();

  // Parse ESBMC options (CMD + set internal options)
  optionst options;
  get_command_line_options(options);

  // for solidity: detect .sol files in positional args or via --sol
  {
    bool is_solidity = cmdline.isset("sol");
    if (!is_solidity)
    {
      for (const auto &arg : cmdline.args)
      {
        if (arg.size() >= 4 && arg.substr(arg.size() - 4) == ".sol")
        {
          is_solidity = true;
          break;
        }
      }
    }
    if (is_solidity)
    {
      options.set_option(
        "no-align-check", true); // no need to check alignment in solidity
      options.set_option("no-unlimited-scanf-check", true);
      options.set_option(
        "force-malloc-success", true); // for calloc in the 'newexpression'

      // Auto-select the best SMT backend for Solidity when the user did not
      // explicitly ask for one. Z3 is significantly slower than modern QF_BV
      // engines on the 256-bit bit-vector arithmetic pervasive in Solidity
      // (uint256, mappings, etc.), so prefer Bitwuzla / CVC5 / Boolector.
      //
      // Exception: k-induction and incremental-bmc issue many incremental
      // queries where Z3 has historically been more robust than CVC5; keep
      // the default (Z3) there so existing regression tests do not regress.
      const bool incremental_mode =
        cmdline.isset("k-induction") || cmdline.isset("k-induction-parallel") ||
        cmdline.isset("incremental-bmc") || cmdline.isset("falsification");
      const bool user_picked_solver =
        cmdline.isset("z3") || cmdline.isset("cvc5") ||
        cmdline.isset("bitwuzla") || cmdline.isset("boolector") ||
        cmdline.isset("yices") || cmdline.isset("mathsat") ||
        cmdline.isset("cvc4") || cmdline.isset("smtlib") ||
        cmdline.isset("neurosym") || cmdline.isset("default-solver");
      if (!user_picked_solver && !incremental_mode)
      {
        const std::string padded =
          std::string(" ") + ESBMC_AVAILABLE_SOLVERS + " ";
        const char *preferred[] = {"bitwuzla", "cvc5", "boolector", "z3"};
        const char *chosen = nullptr;
        for (const char *name : preferred)
        {
          if (padded.find(std::string(" ") + name + " ") != std::string::npos)
          {
            chosen = name;
            break;
          }
        }
        if (chosen)
        {
          options.set_option("default-solver", chosen);
          log_status(
            "Solidity: auto-selecting '{}' as SMT backend (Z3 is much "
            "slower on 256-bit bit-vector arithmetic). Override with "
            "--z3 / --cvc5 / --bitwuzla / --boolector or --default-solver.",
            chosen);
        }
      }
    }
  }

  // Create and preprocess a GOTO program
  if (get_goto_program(options, goto_functions))
    return 6;

  // Output claims about this program
  // (Fedor: should be moved to the output method perhaps)
  if (cmdline.isset("show-claims"))
  {
    const namespacet ns(context);
    show_claims(ns, goto_functions);
    return 0;
  }

  // Set user-specified claims
  // (Fedor: should be moved to the preprocessing method perhaps)
  if (set_claims(goto_functions))
    return 7;

  // Leave without doing any Bounded Model Checking
  if (options.get_bool_option("skip-bmc"))
    return 0;

  // Now run one of the chosen strategies
  if (
    cmdline.isset("termination") || cmdline.isset("incremental-bmc") ||
    cmdline.isset("falsification") || cmdline.isset("k-induction") ||
    cmdline.isset("loop-invariant"))
    return do_bmc_strategy(options, goto_functions);

  // If no strategy is chosen, just rely on the simplifier
  // and the flags set through CMD
  bmct bmc(goto_functions, options, context);
  return do_bmc(bmc);
}

bool esbmc_parseoptionst::resolve_color_option() const
{
  const char *raw = cmdline.getval("color");
  std::string val = (raw && *raw) ? raw : "auto";
  if (val != "auto" && val != "always" && val != "never")
  {
    log_error(
      "Invalid value for --color: '{}'. Must be auto, always, or never.", val);
    exit(1);
  }
  return ENABLE_COLOR(val);
}

// Colorize --flag references found in description text with bold formatting.
// Matches "--" followed by one or more alphanumeric/hyphen characters,
// stopping at delimiters like '.', ',', ' ', ')', '\'', '"', or end of string.
static std::string colorize_flag_refs(const std::string &text)
{
  std::string result;
  size_t i = 0;
  while (i < text.size())
  {
    if (
      i + 2 < text.size() && text[i] == '-' && text[i + 1] == '-' &&
      (std::isalnum((unsigned char)text[i + 2]) || text[i + 2] == '-'))
    {
      size_t start = i;
      i += 2;
      while (i < text.size() &&
             (std::isalnum((unsigned char)text[i]) || text[i] == '-'))
        i++;
      result += CLR_BOLD;
      result += text.substr(start, i - start);
      result += CLR_RESET;
    }
    else
    {
      result += text[i];
      i++;
    }
  }
  return result;
}

// This prints the ESBMC version and a list of CMD options
// available in ESBMC.
void esbmc_parseoptionst::help()
{
  // Redirect everything here to stdout
  FILE *outstream = messaget::state.out;
  messaget::state.out = stdout;

  bool use_color = resolve_color_option();

  // Print the "* * *     ESBMC x.y.z     * * *"
  auto const esbmc_string = fmt::format(" ESBMC {} ", ESBMC_VERSION);
  auto const title_start = std::string("* * * ");
  auto const title_end = std::string(" * * *");
  // Clamp at 0 so an over-long version string can't underflow the
  // unsigned padding width.
  auto const inner = std::max<ptrdiff_t>(
    0,
    80 - static_cast<ptrdiff_t>(
           title_start.length() + title_end.length() + esbmc_string.length()));
  auto const left_pad = std::string(inner / 2, '=');
  auto const right_pad = std::string(inner - inner / 2, '=');
  log_status(
    "\n{}{}{}{}{}", title_start, left_pad, esbmc_string, right_pad, title_end);

  std::ostringstream oss;
  oss << cmdline.cmdline_options;

  if (!use_color)
  {
    log_status("{}", oss.str());
    return;
  }

  // Colorize: group headers in bold cyan, option names in bold,
  // and --flag references in descriptions in bold
  std::istringstream iss(oss.str());
  std::string line;
  while (std::getline(iss, line))
  {
    if (!line.empty() && line[0] != ' ' && line.back() == ':')
      // Group header (e.g. "Printing options:")
      fmt::print(messaget::state.out, CLR_BOLD_CYAN "{}" CLR_RESET "\n", line);
    else if (
      line.size() >= 3 && line[0] == ' ' && line[1] == ' ' && line[2] == '-')
    {
      // Option line: colorize the flag portion (up to the description)
      auto desc_pos = line.find("  ", 4);
      if (desc_pos != std::string::npos)
        fmt::print(
          messaget::state.out,
          CLR_BOLD "{}" CLR_RESET "{}\n",
          line.substr(0, desc_pos),
          colorize_flag_refs(line.substr(desc_pos)));
      else
        fmt::print(messaget::state.out, CLR_BOLD "{}" CLR_RESET "\n", line);
    }
    else
      fmt::print(messaget::state.out, "{}\n", colorize_flag_refs(line));
  }

  // Restore everything back to original output stream.
  messaget::state.out = outstream;
}
