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
#include <goto-programs/dead_store_analysis.h>
#include <util/cwe_mapping.h>
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

#define BT_BUF_SIZE 256

// ANSI color/style escape sequences for terminal output
#define CLR_BOLD_CYAN "\033[1;36m"
#define CLR_BOLD "\033[1m"
#define CLR_RESET "\033[0m"

extern "C" const char buildidstring_buf[];
extern "C" const unsigned int buildidstring_buf_size;

static std::string_view esbmc_version_string()
{
  return {buildidstring_buf, buildidstring_buf_size};
}

enum PROCESS_TYPE
{
  BASE_CASE,
  FORWARD_CONDITION,
  INDUCTIVE_STEP,
  NUM_CHILD_PROCESSES,
  PARENT = NUM_CHILD_PROCESSES
};

struct resultt
{
  PROCESS_TYPE type;
  uint64_t k;
};

#ifndef _WIN32
void timeout_handler(int)
{
  log_error("Timed out");
  // Kill any external solver process groups first: they are in their own
  // groups, so they outlive this _exit() otherwise (e.g. an mpirun job).
  file_operations::kill_registered_pgroups();
  file_operations::cleanup_registered_tmps();
  // Use _exit to avoid atexit handlers that may deadlock the allocator
  _exit(1);
}
#endif

#ifndef _WIN32
/* This will produce output on stderr that looks somewhat like this:
 *   Signal 6, backtrace:
 *   src/esbmc/esbmc(+0xad52e)[0x556c5dcdb52e]
 *   /lib64/libc.so.6(+0x39d50)[0x7f7a8f475d50]
 *   /lib64/libc.so.6(+0x89d9c)[0x7f7a8f4c5d9c]
 *   /lib64/libc.so.6(raise+0x12)[0x7f7a8f475ca2]
 *   /lib64/libc.so.6(abort+0xd3)[0x7f7a8f45e4ed]
 *   src/esbmc/esbmc(+0x62e3e5)[0x556c5e25c3e5]
 *   src/esbmc/esbmc(+0x61f7f1)[0x556c5e24d7f1]
 *   [...]
 *
 *   Memory map:
 *   [...]
 *
 * The backtrace can be translated into proper function symbols via addr2line,
 * e.g.
 *
 *   cat bt | tr -d '[]' | tr '()' ' ' | grep esbmc | \
 *   while read f a b; do echo $a | tr -d '+'; done | \
 *   xargs addr2line -iapfCr -e src/esbmc/esbmc
 */
static void segfault_handler(int sig)
{
  ::signal(sig, SIG_DFL);
  void *buffer[BT_BUF_SIZE];
#  ifdef __GLIBC__
  int n = backtrace(buffer, BT_BUF_SIZE);
  dprintf(STDERR_FILENO, "\nSignal %d, backtrace:\n", sig);
  backtrace_symbols_fd(buffer, n, STDERR_FILENO);
#  endif
  int fd = open("/proc/self/maps", O_RDONLY);
  if (fd != -1)
  {
    dprintf(STDERR_FILENO, "\nMemory map:\n");
    // Bounded read: `buffer` is a fixed-size stack array and the loop
    // passes its exact size to read(2), so no overflow is possible.
    // Loop terminates on EOF (rd == 0) or on a non-EINTR error.
    for (ssize_t rd; (rd = read(fd, buffer, sizeof(buffer))) > 0 ||
                     (rd == -1 && errno == EINTR);)
      rd = write(STDERR_FILENO, buffer, rd < 0 ? 0 : rd);
    close(fd);
  }
  ::raise(sig);
}
#endif

// This transforms a string representation of a time interval
// written in the form <number><suffix> into seconds.
// The following suffixes corresponding to time units are supported:
//
//  s - seconds,
//  m - minutes,
//  h - hours,
//  d - days.
//
// When <suffix> is empty, the default time unit is seconds.
// If <suffix> is not empty, and its final character is not in the list above,
// this method throws an error.
//
// \param str - string representation of a time interval,
// \return - number of seconds that represents the input string value.
uint64_t esbmc_parseoptionst::read_time_spec(std::string_view str)
{
  if (str.empty())
  {
    log_error("Empty timeout value");
    abort();
  }
  uint64_t mult = 1;
  if (!isdigit((unsigned char)str.back()))
  {
    switch (str.back())
    {
    case 's':
      mult = 1;
      break;
    case 'm':
      mult = 60;
      break;
    case 'h':
      mult = 3600;
      break;
    case 'd':
      mult = 86400;
      break;
    default:
      log_error("Unrecognized timeout suffix");
      abort();
    }
  }
  uint64_t timeout = 0;
  std::from_chars(str.data(), str.data() + str.size(), timeout);
  timeout *= mult;
  return timeout;
}

// This transforms a string representation of a memory limit
// written in the form <number><suffix> into megabytes.
// The following suffixes corresponding to memory size units are supported:
//
//  b - bytes,
//  k - kilobytes,
//  m - megabytes,
//  g - gigabytes.
//
// When <suffix> is empty, the default unit is megabytes.
// If <suffix> is not empty, and its final character is not in the list above,
// this method throws an error.
//
// \param str - string representation of a memory limit,
// \return - number of bytes that represents the input string value.
uint64_t esbmc_parseoptionst::read_mem_spec(std::string_view str)
{
  if (str.empty())
  {
    log_error("Empty memlimit value");
    abort();
  }
  uint64_t mult = 1024ULL * 1024ULL;
  if (!isdigit((unsigned char)str.back()))
  {
    switch (str.back())
    {
    case 'b':
      mult = 1;
      break;
    case 'k':
      mult = 1024;
      break;
    case 'm':
      mult = 1024ULL * 1024ULL;
      break;
    case 'g':
      mult = 1024ULL * 1024ULL * 1024ULL;
      break;
    default:
      log_error("Unrecognized memlimit suffix");
      abort();
    }
  }
  uint64_t size = 0;
  std::from_chars(str.data(), str.data() + str.size(), size);
  size *= mult;
  return size;
}

static std::string format_target()
{
  const char *endian = nullptr;
  switch (config.ansi_c.endianess)
  {
  case configt::ansi_ct::IS_LITTLE_ENDIAN:
    endian = "little";
    break;
  case configt::ansi_ct::IS_BIG_ENDIAN:
    endian = "big";
    break;
  case configt::ansi_ct::NO_ENDIANESS:
    endian = "no";
    break;
  }
  assert(endian);
  const char *lib = nullptr;
  switch (config.ansi_c.lib)
  {
  case configt::ansi_ct::LIB_NONE:
    lib = "system";
    break;
  case configt::ansi_ct::LIB_FULL:
    lib = "esbmc";
    break;
  }
  assert(lib);
  std::ostringstream oss;
  oss << config.ansi_c.word_size << "-bit " << endian << "-endian "
      << config.ansi_c.target.to_string() << " with " << lib << "libc";
  return oss.str();
}

// This method creates a set of options based on the CMD arguments passed to
// ESBMC. Also, it sets some options that are used across various
// ESBMC stages but which are not available via CMD.
//
// \param options - the options object created and updated by this method.
void esbmc_parseoptionst::get_command_line_options(optionst &options)
{
  if (config.set(cmdline))
    exit(1);

  log_status("Target: {}", format_target());

  // Copy all flags that are set to non-default values in CMD into options
  options.cmdline(cmdline);
  set_verbosity_msg();

  // Resolve --color option: validate and convert to boolean
  options.set_option("color", resolve_color_option());

  if (cmdline.isset("git-hash"))
  {
    log_result("{}", esbmc_version_string());
    exit(0);
  }

  if (cmdline.isset("list-solvers"))
  {
    // Generated for us by autoconf,
    log_result("Available solvers: {}", ESBMC_AVAILABLE_SOLVERS);
    exit(0);
  }

  // Below we make some additional adjustments (e.g., adding some options
  // that are used by ESBMC at later stages but which are not available
  // through CMD, setting groups of options based depending on
  // particular CMD flags)
  if (cmdline.isset("bv"))
    options.set_option("int-encoding", false);

  if (cmdline.isset("ir"))
    options.set_option("int-encoding", true);

  if (cmdline.isset("ir-ieee"))
  {
    options.set_option("int-encoding", true);
    options.set_option("ir-ieee", true);
  }

  // --ir requests integer/real arithmetic encoding via the SMT Int sort.
  // Bitwuzla and Boolector are bit-vector-only backends; pairing them with
  // --ir silently produces wrong-answer behaviour at solve time. cvc4, cvc5,
  // yices, and mathsat all support Int and are left alone.
  if (cmdline.isset("ir") || cmdline.isset("ir-ieee"))
  {
    for (const char *s : {"bitwuzla", "boolector"})
    {
      if (cmdline.isset(s))
      {
        log_error(
          "--{} requires a solver that supports integer/real arithmetic. "
          "--{} only supports bit-vector arithmetic. Re-run without --{}, "
          "or drop --{} (--ir defaults to Z3).",
          cmdline.isset("ir-ieee") ? "ir-ieee" : "ir",
          s,
          s,
          s);
        exit(1);
      }
    }
  }
  if (cmdline.isset("fixedbv"))
    options.set_option("fixedbv", true);
  else
    options.set_option("floatbv", true);

  if (cmdline.isset("context-bound"))
    options.set_option("context-bound", cmdline.getval("context-bound"));
  else
    options.set_option("context-bound", -1);

  if (cmdline.isset("deadlock-check"))
  {
    options.set_option("deadlock-check", true);
    options.set_option("atomicity-check", false);
  }
  else
    options.set_option("deadlock-check", false);

  if (cmdline.isset("compact-trace"))
    options.set_option("no-slice", true);

  if (
    cmdline.isset("smt-thread-guard") || cmdline.isset("smt-symex-guard") ||
    cmdline.isset("smt-symex-assert") || cmdline.isset("smt-symex-assume"))
  {
    log_status(
      "Enabling --smt-during-symex to use features that involve encoding SMT "
      "during symex");
    options.set_option("smt-during-symex", true);
  }

  // check the user's parameters to run incremental verification
  if (!cmdline.isset("unlimited-k-steps"))
  {
    // Get max number of iterations
    uint64_t max_k_step = strtoul(cmdline.getval("max-k-step"), nullptr, 10);

    // Get the increment
    uint64_t k_step_inc = strtoul(cmdline.getval("k-step"), nullptr, 10);

    // Get the start of the base-case, default 1
    uint64_t k_step_base = strtoul(cmdline.getval("base-k-step"), nullptr, 10);

    // check whether k-step is greater than max-k-step
    if (k_step_inc >= max_k_step)
    {
      log_error(
        "Please specify --k-step smaller than max-k-step if you want to use "
        "incremental verification.");
      abort();
    }

    // check whether k_step_inc is greater than max-k-step
    if (k_step_base >= max_k_step)
    {
      log_error(
        "Please specify --base-k-step smaller than max-k-step if you want "
        "to use incremental verification.");
      abort();
    }
  }

  if (cmdline.isset("base-case"))
  {
    options.set_option("base-case", true);
    options.set_option("no-unwinding-assertions", true);
    options.set_option("partial-loops", false);
  }

  if (cmdline.isset("forward-condition"))
  {
    options.set_option("forward-condition", true);
    options.set_option("no-unwinding-assertions", false);
    options.set_option("partial-loops", false);
    options.set_option("no-assertions", true);
  }

  if (cmdline.isset("inductive-step"))
  {
    options.set_option("inductive-step", true);
    options.set_option("no-unwinding-assertions", true);
    options.set_option("partial-loops", false);
  }

  if (cmdline.isset("validate-correctness-witness"))
  {
    if (!cmdline.isset("witness"))
    {
      log_error(
        "--validate-correctness-witness requires --witness <file.yaml>");
      abort();
    }
    const std::string witness = cmdline.getval("witness");
    const boost::filesystem::path wp(witness);
    if (!boost::filesystem::exists(wp))
    {
      log_error("Witness file '{}' does not exist.", witness);
      abort();
    }
    if (wp.extension() != ".yaml" && wp.extension() != ".yml")
    {
      log_error(
        "Witness file has extension {}, expected yaml or yml.",
        wp.extension().string());
      abort();
    }
    options.set_option("validate-correctness-witness", true);
    options.set_option("witness", witness);
  }

  if (cmdline.isset("validate-violation-witness"))
  {
    if (!cmdline.isset("witness"))
    {
      log_error("--validate-violation-witness requires --witness <file.yaml>");
      abort();
    }
    const std::string witness = cmdline.getval("witness");
    const boost::filesystem::path wp(witness);
    if (!boost::filesystem::exists(wp))
    {
      log_error("Witness file '{}' does not exist.", witness);
      abort();
    }
    if (wp.extension() != ".yaml" && wp.extension() != ".yml")
    {
      log_error(
        "Witness file has extension {}, expected yaml or yml.",
        wp.extension().string());
      abort();
    }
    options.set_option("validate-violation-witness", true);
    options.set_option("witness", witness);
  }

  // --loop-invariant implicitly enables k-induction solving so that
  // do_bmc_strategy runs the full base/forward/inductive-step loop.
  if (
    cmdline.isset("loop-invariant") ||
    cmdline.isset("validate-correctness-witness"))
    options.set_option("k-induction", true);

  // The IS pointer-invariant work (symex_assign / symex_dereference)
  // only kicks in when --add-symex-value-sets is enabled, and the
  // SV-COMP wrapper has been setting it for k-induction runs all
  // along. Mirror that default for direct CLI users so they get the
  // same IS encoding (and the same proofs of pointer-traversing
  // loops) without needing to know about the flag.
  if (
    cmdline.isset("k-induction") || cmdline.isset("k-induction-parallel") ||
    cmdline.isset("inductive-step"))
    options.set_option("add-symex-value-sets", true);

  // Default-enable the vacuity probe under --loop-invariant-check (the
  // standalone Hoare-rewrite mode). A loop invariant that implies the guard
  // makes the post-loop continuation unreachable; without this probe every
  // downstream claim discharges as vacuously true.
  //
  // We deliberately do NOT default-enable for combined mode --loop-invariant:
  // that runs k-induction phases (base case, forward condition, inductive
  // step) whose UNSAT-on-internal-claims is the success signal, not vacuity.
  // Users can opt in explicitly with --check-vacuity in those modes.
  if (cmdline.isset("no-vacuity-check"))
    options.set_option("check-vacuity", false);
  else if (
    cmdline.isset("check-vacuity") || cmdline.isset("loop-invariant-check"))
    options.set_option("check-vacuity", true);

  // Check for conflicting strategies
  if (cmdline.isset("k-induction") && cmdline.isset("termination"))
  {
    log_warning(
      "Both --k-induction and --termination specified. "
      "Using --k-induction (which does not include termination checking).");
    // Optionally disable termination flag
    options.set_option("termination", false);
  }

  // interval-symex-guard is designed for plain BMC loop-counter tracking.
  // Disable it for advanced verification modes whose GOTO/symex transformations
  // are incompatible with a single shared (non-forked) interval_domaint:
  //   - incremental-BMC reuses one goto_symext across unwind iterations
  //   - k-induction (base/forward/inductive) havocs loop variables
  //   - loop-invariant and function contracts inject havoc+assume sequences
  if (
    options.get_bool_option("k-induction") ||
    cmdline.isset("k-induction-parallel") || cmdline.isset("incremental-bmc") ||
    cmdline.isset("termination") || cmdline.isset("enforce-contract") ||
    cmdline.isset("enforce-all-contracts") ||
    cmdline.isset("replace-call-with-contract") ||
    cmdline.isset("replace-all-contracts") || cmdline.isset("base-case") ||
    cmdline.isset("forward-condition") || cmdline.isset("inductive-step"))
    options.set_option("no-interval-symex-guard", true);

  if (
    cmdline.isset("overflow-check") || cmdline.isset("unsigned-overflow-check"))
    options.set_option("disable-inductive-step", true);

  if (cmdline.isset("ub-shift-check"))
    options.set_option("ub-shift-check", true);

  if (cmdline.isset("clz-zero-check"))
    options.set_option("clz-zero-check", true);

  if (cmdline.isset("timeout"))
  {
#ifdef _WIN32
    log_error("Timeout unimplemented on Windows, sorry");
    abort();
#else
    const char *time = cmdline.getval("timeout");
    uint64_t timeout = read_time_spec(time);
    signal(SIGALRM, timeout_handler);
    alarm(timeout);
#endif
  }

  if (cmdline.isset("memlimit"))
  {
#ifdef _WIN32
    log_error("Can't memlimit on Windows, sorry");
    abort();
#else
    uint64_t size = read_mem_spec(cmdline.getval("memlimit"));

    struct rlimit lim;
    lim.rlim_cur = size;
    lim.rlim_max = size;
    if (setrlimit(RLIMIT_DATA, &lim) != 0)
    {
      perror("Couldn't set memory limit");
      abort();
    }
#endif
  }

#ifndef _WIN32
  struct rlimit lim;
  if (cmdline.isset("enable-core-dump"))
  {
    lim.rlim_cur = RLIM_INFINITY;
    lim.rlim_max = RLIM_INFINITY;
    if (setrlimit(RLIMIT_CORE, &lim) != 0)
    {
      perror("Couldn't unlimit core dump size");
      abort();
    }
  }
  else
  {
    lim.rlim_cur = 0;
    lim.rlim_max = 0;
    if (setrlimit(RLIMIT_CORE, &lim) != 0)
    {
      perror("Couldn't disable core dump size");
      abort();
    }
  }
#endif

#ifndef _WIN32
  if (cmdline.isset("segfault-handler"))
  {
    signal(SIGSEGV, segfault_handler);
    signal(SIGABRT, segfault_handler);
  }
#endif

  // parallel solving activates "--multi-property"
  if (cmdline.isset("parallel-solving"))
  {
    options.set_option("base-case", true);
    options.set_option("multi-property", true);
  }

  // --all-witnesses also activates --multi-property
  if (cmdline.isset("all-witnesses"))
  {
    if (cmdline.isset("max-witnesses"))
    {
      int max_w = std::stoi(cmdline.getval("max-witnesses"));
      if (max_w < 0)
      {
        log_error("--max-witnesses must be >= 0 (got {})", max_w);
        abort();
      }
    }

    const bool was_multi = options.get_bool_option("multi-property") ||
                           cmdline.isset("multi-property");
    if (!was_multi)
      log_status("--all-witnesses: auto-enabling --multi-property");
    options.set_option("multi-property", true);
    // Don't disturb base-case if the user explicitly picked a different
    // k-induction phase (forward-condition-only or inductive-step-only).
    if (!cmdline.isset("forward-condition") && !cmdline.isset("inductive-step"))
      options.set_option("base-case", true);
  }

  // If multi-property is on, we should set base-case
  if (cmdline.isset("multi-property"))
  {
    options.set_option("base-case", true);
  }

  /* compatibility: --cvc maps to --cvc4 */
  if (cmdline.isset("cvc"))
    options.set_option("cvc4", true);

  if (cmdline.isset("log-message"))
    options.set_option("log-message", true);

  if (cmdline.isset("keep_alive_running"))
    options.set_option("keep_alive_running", true);

  if (cmdline.isset("keep-alive-interval"))
    options.set_option(
      "keep-alive-interval", cmdline.getval("keep-alive-interval"));

  if (cmdline.isset("override-return-annotation"))
    options.set_option("override-return-annotation", true);

  if (cmdline.isset("witness-output-yaml"))
  {
    std::string filename = cmdline.getval("witness-output-yaml");
    boost::filesystem::path n(filename);

    if (n.extension() == ".yaml" || n.extension() == ".yml")
    {
      // expected extension
    }
    else if (!n.has_extension())
    {
      if (n != "-")
        options.set_option("witness-output-yaml", filename + ".yml");
    }
    else
    {
      log_error(
        "Output file has extension {}, expected yaml or yml.",
        n.extension().string());
      abort();
    }
  }

  if (cmdline.isset("witness-output-graphml"))
  {
    std::string filename = cmdline.getval("witness-output-graphml");
    boost::filesystem::path n(filename);

    if (n.extension() == ".graphml")
    {
      // expected extension
    }
    else if (!n.has_extension())
    {
      if (n != "-")
        options.set_option("witness-output-graphml", filename + ".graphml");
    }
    else
    {
      log_error(
        "Output file has extension {}, expected graphml.",
        n.extension().string());
      abort();
    }
  }

  if (cmdline.isset("witness-output"))
  {
    std::string filename = cmdline.getval("witness-output");
    boost::filesystem::path n(filename);
    n.replace_extension("");

    options.set_option("witness-output-yaml", filename + ".yml");
    options.set_option("witness-output-graphml", filename + ".graphml");
  }

  if (cmdline.isset("sarif-output"))
    options.set_option("sarif-output", cmdline.getval("sarif-output"));

  // Fail fast if the user explicitly requested an SMT solver that is not
  // built into this ESBMC binary, before spending time parsing the program.
  check_solver_availability(options);

  config.options = options;
}

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
  if (cmdline.isset("k-induction-parallel"))
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
        cmdline.isset("default-solver");
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

// This is the parallel version of k-induction algorithm.
// This is an old implementation and should be revisited sometime in the
// future.
int esbmc_parseoptionst::doit_k_induction_parallel()
{
#ifdef _WIN32
  log_error("Windows does not support parallel kind");
  abort();
#else
  // Pipes for communication between processes
  int forward_pipe[2], backward_pipe[2];

  // Process type
  PROCESS_TYPE process_type = PARENT;

  if (pipe(forward_pipe))
  {
    log_status("\nPipe Creation Failed, giving up.");
    _exit(1);
  }

  if (pipe(backward_pipe))
  {
    log_status("\nPipe Creation Failed, giving up.");
    _exit(1);
  }

  /* Set file descriptor non-blocking */
  fcntl(
    backward_pipe[0], F_SETFL, fcntl(backward_pipe[0], F_GETFL) | O_NONBLOCK);

  pid_t children_pid[3];
  short num_p = 0;

  // We need to fork 3 times: one for each step
  for (unsigned p = 0; p < 3; ++p)
  {
    pid_t pid = fork();

    if (pid == -1)
    {
      log_status("\nFork Failed, giving up.");
      _exit(1);
    }

    // Child process
    if (!pid)
    {
      process_type = PROCESS_TYPE(p);
      break;
    }
    // Parent process

    children_pid[p] = pid;
    ++num_p;
  }

  if (process_type == PARENT && num_p != 3)
  {
    log_error("Child processes were not created sucessfully.");
    abort();
  }

  optionst options;

  if (process_type != PARENT)
  {
    // Get full set of options
    get_command_line_options(options);

    // Generate goto functions and set claims
    if (get_goto_program(options, goto_functions))
      return 6;

    if (cmdline.isset("show-claims"))
    {
      const namespacet ns(context);
      show_claims(ns, goto_functions);
      return 0;
    }

    if (set_claims(goto_functions))
      return 7;
  }

  // Get max number of iterations
  uint64_t max_k_step = cmdline.isset("unlimited-k-steps")
                          ? UINT_MAX
                          : strtoul(cmdline.getval("max-k-step"), nullptr, 10);

  // Get the increment
  uint64_t k_step_inc = strtoul(cmdline.getval("k-step"), nullptr, 10);

  // Get the start of the base-case, default 1
  uint64_t k_step_base = strtoul(cmdline.getval("base-k-step"), nullptr, 10);
  if (k_step_base >= max_k_step)
  {
    log_error(
      "Please specify --base-k-step smaller than max-k-step if you want "
      "to use incremental verification.");
    abort();
  }

  // All processes were created successfully
  switch (process_type)
  {
  case PARENT:
  {
    // Communication to child processes
    close(forward_pipe[1]);
    close(backward_pipe[0]);

    struct resultt a_result;
    bool finished[NUM_CHILD_PROCESSES] = {};
    bool intentionally_killed[NUM_CHILD_PROCESSES] = {};
    const char *process_name[NUM_CHILD_PROCESSES] = {
      "base case", "forward condition", "inductive step"};
    uint64_t solution[NUM_CHILD_PROCESSES] = {
      max_k_step, max_k_step, max_k_step};

    // Keep reading until we find an answer
    while (
      !(finished[BASE_CASE] && finished[FORWARD_CONDITION] &&
        finished[INDUCTIVE_STEP]))
    {
      // Bounded read: destination is a single resultt on the stack and
      // the read length is its exact sizeof. Short reads (EOF, error,
      // EAGAIN) are checked explicitly below.
      bool valid_read = true;
      int read_size = read(forward_pipe[0], &a_result, sizeof(resultt));
      if (read_size != sizeof(resultt))
      {
        if (read_size == 0)
        {
          // Client hung up; check child status but don't interpret result.
          valid_read = false;
        }
        else
        {
          // Invalid size read.
          log_error("Short read communicating with kinduction children");
          log_error("Size {}, expected {}", read_size, sizeof(resultt));
          abort();
        }
      }

      // Check if any child process has terminated
      for (int i = 0; i < NUM_CHILD_PROCESSES; i++)
      {
        if (finished[i])
          continue;

        int status;
        pid_t result = waitpid(children_pid[i], &status, WNOHANG);
        if (result <= 0)
          continue;

        if (intentionally_killed[i] || WIFEXITED(status))
        {
          finished[i] = true;
        }
        else if (WIFSIGNALED(status))
        {
          log_warning(
            "{} process was terminated by signal {:d}.",
            process_name[i],
            WTERMSIG(status));
          std::fill(finished, finished + NUM_CHILD_PROCESSES, true);
        }
      }

      if (!valid_read)
        continue;

      switch (a_result.type)
      {
      case BASE_CASE:
      case FORWARD_CONDITION:
      case INDUCTIVE_STEP:
        finished[a_result.type] = true;
        solution[a_result.type] = a_result.k;
        break;

      default:
        log_error("Message from unrecognized k-induction child process");
        abort();
      }

      // If either the base case found a bug or the forward condition
      // finds a solution, present the result
      if (
        finished[BASE_CASE] && (solution[BASE_CASE] != 0) &&
        (solution[BASE_CASE] != max_k_step))
        break;

      // If the either the forward condition or inductive step finds a
      // solution, first check if base case couldn't find a bug in that code,
      // if there is no bug, inductive step can present the result
      if (
        finished[FORWARD_CONDITION] && (solution[FORWARD_CONDITION] != 0) &&
        (solution[FORWARD_CONDITION] != max_k_step))
      {
        // If base case finished, then we can present the result
        if (finished[BASE_CASE])
          break;

        // Otherwise, kill the inductive step process
        intentionally_killed[INDUCTIVE_STEP] = true;
        kill(children_pid[INDUCTIVE_STEP], SIGKILL);

        // And ask base case for a solution

        // Struct to keep the result
        struct resultt r = {process_type, 0};

        r.k = solution[FORWARD_CONDITION];

        // Write result
        auto const len = write(backward_pipe[1], &r, sizeof(r));
        assert(len == sizeof(r) && "short write");
        (void)len; //ndebug
      }

      else if (
        finished[INDUCTIVE_STEP] && (solution[INDUCTIVE_STEP] != 0) &&
        (solution[INDUCTIVE_STEP] != max_k_step))
      {
        // If base case finished, then we can present the result
        if (finished[BASE_CASE])
          break;

        // Otherwise, kill the forward condition process
        intentionally_killed[FORWARD_CONDITION] = true;
        kill(children_pid[FORWARD_CONDITION], SIGKILL);

        // And ask base case for a solution

        // Struct to keep the result
        struct resultt r = {process_type, 0};

        r.k = solution[INDUCTIVE_STEP];

        // Write result
        auto const len = write(backward_pipe[1], &r, sizeof(r));
        assert(len == sizeof(r) && "short write");
        (void)len; //ndebug
      }
    }

    for (int i : children_pid)
      kill(i, SIGKILL);

    // Check if a solution was found by the base case
    if (
      finished[BASE_CASE] && (solution[BASE_CASE] != 0) &&
      (solution[BASE_CASE] != max_k_step))
    {
      log_result(
        "\nBug found by the base case (k = {})\nVERIFICATION FAILED",
        solution[BASE_CASE]);
      return true;
    }

    // Check if a solution was found by the forward condition
    if (
      finished[FORWARD_CONDITION] && (solution[FORWARD_CONDITION] != 0) &&
      (solution[FORWARD_CONDITION] != max_k_step))
    {
      // We should only present the result if the base case finished
      // and haven't crashed (if it crashed, solution will be UINT_MAX)
      if (finished[BASE_CASE] && (solution[BASE_CASE] != max_k_step))
      {
        log_success(
          "\nSolution found by the forward condition; "
          "all states are reachable (k = {:d})\n"
          "VERIFICATION SUCCESSFUL",
          solution[FORWARD_CONDITION]);
        return false;
      }
    }

    // Check if a solution was found by the inductive step
    if (
      finished[INDUCTIVE_STEP] && (solution[INDUCTIVE_STEP] != 0) &&
      (solution[INDUCTIVE_STEP] != max_k_step))
    {
      // We should only present the result if the base case finished
      // and haven't crashed (if it crashed, solution will be UINT_MAX)
      if (finished[BASE_CASE] && (solution[BASE_CASE] != max_k_step))
      {
        log_success(
          "\nSolution found by the inductive step "
          "(k = {:d})\n"
          "VERIFICATION SUCCESSFUL",
          solution[INDUCTIVE_STEP]);
        return false;
      }
    }

    // Couldn't find a bug or a proof for the current depth
    log_fail("\nVERIFICATION UNKNOWN");
    return false;
  }

  case BASE_CASE:
  {
    // Set that we are running base case
    options.set_option("base-case", true);
    options.set_option("forward-condition", false);
    options.set_option("inductive-step", false);

    options.set_option("no-unwinding-assertions", true);
    options.set_option("partial-loops", false);

    // Start communication to the parent process
    close(forward_pipe[0]);
    close(backward_pipe[1]);

    // Struct to keep the result
    struct resultt r = {process_type, 0};

    // Run bmc and only send results in two occasions:
    // 1. A bug was found, we send the step where it was found
    // 2. It couldn't find a bug
    for (uint64_t k_step = k_step_base; k_step <= max_k_step;
         k_step += k_step_inc)
    {
      bmct bmc(goto_functions, options, context);
      bmc.options.set_option("unwind", integer2string(k_step));

      log_progress("Checking base case, k = {:d}\n", k_step);

      // If an exception was thrown, we should abort the process
      smt_resultt res = P_ERROR;
      try
      {
        res = static_cast<smt_resultt>(do_bmc(bmc));
      }
      catch (...)
      {
        break;
      }

      // Send information to parent if no bug was found
      if (res == P_SATISFIABLE)
      {
        r.k = k_step;

        // Write result
        auto const len = write(forward_pipe[1], &r, sizeof(r));
        assert(len == sizeof(r) && "short write");
        (void)len; //ndebug

        log_status("Base case process finished (bug found).\n");
        return true;
      }

      // Check if the parent process is asking questions

      // Bounded read: destination is a single resultt on the stack and
      // the read length is its exact sizeof. Short reads (EOF, error,
      // EAGAIN) are checked explicitly below.
      struct resultt a_result;
      int read_size = read(backward_pipe[0], &a_result, sizeof(resultt));
      if (read_size != sizeof(resultt))
      {
        if (read_size == 0)
        {
          // Client hung up; continue on, but don't interpret the result.
          continue;
        }
        if (read_size == -1 && errno == EAGAIN)
        {
          // No data available yet
          continue;
        }
        else
        {
          // Invalid size read.
          log_error("Short read communicating with kinduction parent");
          log_error("Size {}, expected {}", read_size, sizeof(resultt));

          abort();
        }
      }

      // We only receive messages from the parent
      assert(a_result.type == PARENT);

      // If the value being asked is greater or equal the current step,
      // then we can stop the base case. It can be equal, because we
      // have just checked the current value of k
      if (a_result.k < k_step)
        break;

      // Otherwise, we just need to check the base case for k = a_result.k
      max_k_step = a_result.k + k_step_inc;
    }

    // Send information to parent that a bug was not found
    r.k = 0;

    auto const len = write(forward_pipe[1], &r, sizeof(r));
    assert(len == sizeof(r) && "short write");
    (void)len; //ndebug

    log_status("Base case process finished (no bug found).\n");
    return false;
  }

  case FORWARD_CONDITION:
  {
    // Set that we are running forward condition
    options.set_option("base-case", false);
    options.set_option("forward-condition", true);
    options.set_option("inductive-step", false);

    options.set_option("no-unwinding-assertions", false);
    options.set_option("partial-loops", false);
    options.set_option("no-assertions", true);

    // Start communication to the parent process
    close(forward_pipe[0]);
    close(backward_pipe[1]);

    // Struct to keep the result
    struct resultt r = {process_type, 0};

    // Run bmc and only send results in two occasions:
    // 1. A proof was found, we send the step where it was found
    // 2. It couldn't find a proof
    for (uint64_t k_step = k_step_base + 1; k_step <= max_k_step;
         k_step += k_step_inc)
    {
      bmct bmc(goto_functions, options, context);
      bmc.options.set_option("unwind", integer2string(k_step));

      log_status("Checking forward condition, k = {:d}", k_step);

      // If an exception was thrown, we should abort the process
      smt_resultt res = P_ERROR;
      try
      {
        res = static_cast<smt_resultt>(do_bmc(bmc));
      }
      catch (...)
      {
        break;
      }

      if (options.get_bool_option("disable-forward-condition"))
        break;

      // Send information to parent if no bug was found
      if (res == P_UNSATISFIABLE)
      {
        r.k = k_step;

        // Write result
        auto const len = write(forward_pipe[1], &r, sizeof(r));
        assert(len == sizeof(r) && "short write");
        (void)len; //ndebug

        log_status("Forward condition process finished (safety proven).");
        return false;
      }
    }

    // Send information to parent that it couldn't prove the code
    r.k = 0;

    auto const len = write(forward_pipe[1], &r, sizeof(r));
    assert(len == sizeof(r) && "short write");
    (void)len; //ndebug

    log_status("Forward condition process finished (safety not proven).");
    return true;
  }

  case INDUCTIVE_STEP:
  {
    // Set that we are running inductive step
    options.set_option("base-case", false);
    options.set_option("forward-condition", false);
    options.set_option("inductive-step", true);

    options.set_option("no-unwinding-assertions", true);
    options.set_option("partial-loops", true);

    // Start communication to the parent process
    close(forward_pipe[0]);
    close(backward_pipe[1]);

    // Struct to keep the result
    struct resultt r = {process_type, 0};

    // Run bmc and only send results in two occasions:
    // 1. A proof was found, we send the step where it was found
    // 2. It couldn't find a proof
    for (uint64_t k_step = k_step_base + 1; k_step <= max_k_step;
         k_step += k_step_inc)
    {
      bmct bmc(goto_functions, options, context);

      bmc.options.set_option("unwind", integer2string(k_step));

      log_status("Checking inductive step, k = {:d}", k_step);

      // If an exception was thrown, we should abort the process
      smt_resultt res = P_ERROR;
      try
      {
        res = static_cast<smt_resultt>(do_bmc(bmc));
      }
      catch (...)
      {
        break;
      }

      if (options.get_bool_option("disable-inductive-step"))
        break;

      // Send information to parent if no bug was found
      if (res == P_UNSATISFIABLE)
      {
        r.k = k_step;

        // Write result
        auto const len = write(forward_pipe[1], &r, sizeof(r));
        assert(len == sizeof(r) && "short write");
        (void)len; //ndebug

        log_status("Inductive process finished (safety proven).");
        return false;
      }
    }

    // Send information to parent that it couldn't prove the code
    r.k = 0;

    auto const len = write(forward_pipe[1], &r, sizeof(r));
    assert(len == sizeof(r) && "short write");
    (void)len; //ndebug

    log_status("Inductive process finished (safety not proven).");
    return true;
  }

  default:
    assert(0 && "Unknown process type.");
  }

#endif

  return 0;
}

// Emit CWE-835 (non-termination / infinite loop) into the structured
// outputs when --termination refutes the termination property.
//
// A non-termination verdict is proven by UNSAT at a loop exit marker, so
// unlike a normal property violation it has no counterexample trace to
// drive the structured emitters. We synthesise a single-step trace
// anchored to a real termination-marker ASSERT instruction: that gives a
// valid `pc` and a source location, and lets the existing emitters reuse
// the util/cwe_mapping single source of truth (@p comment matches the
// "non-terminating execution" rule -> CWE-835).
//
// CWE-835 is a loop weakness, so a per-loop marker is preferred as the
// anchor; an abort-call marker is only a fallback. Markers in library
// helpers (body.hide) rank below both, so the anchor never lands in
// ESBMC's own installed sources. The inductive-step verdict is a
// whole-program property, so with several loops present the chosen
// marker is a best-effort representative location — only the CWE id
// (835) is guaranteed, not that it is the specific non-terminating loop.
static void report_non_termination_cwe(
  optionst &options,
  const namespacet &ns,
  const goto_functionst &goto_functions,
  const std::string &comment)
{
  // The YAML (SV-COMP 2.0) witness format has no CWE field — CWE support is
  // GraphML-only — so it is deliberately not emitted here.
  const bool want_sarif = !options.get_option("sarif-output").empty();
  const bool want_graphml =
    !options.get_option("witness-output-graphml").empty();
  const bool want_json = options.get_bool_option("generate-json-report");
  if (!(want_sarif || want_graphml || want_json))
    return;

  // Anchor candidates, best first. A user-source location always beats a
  // library helper: goto_termination inserts per-loop markers into helpers
  // too (see the marker pass there), and __ESBMC_atexit_handler's
  // `while (atexit_count > 0)` is linked into every program. Since
  // function_map is ordered by mangled id, `c:@F@__ESBMC_atexit_handler`
  // sorts before `c:@F@main`, so scanning without this rank anchors the
  // CWE to ESBMC's own stdlib.c instead of the user's loop.
  enum anchor_rankt
  {
    USER_LOOP = 0,
    USER_ABORT,
    LIB_LOOP,
    LIB_ABORT,
    NO_ANCHOR
  };

  goto_programt::const_targett marker;
  anchor_rankt best = NO_ANCHOR;
  for (const auto &fn : goto_functions.function_map)
  {
    if (!fn.second.body_available)
      continue;
    const bool is_lib = fn.second.body.hide;
    for (auto it = fn.second.body.instructions.begin();
         best != USER_LOOP && it != fn.second.body.instructions.end();
         ++it)
    {
      if (!it->is_assert())
        continue;
      const std::string mc = it->location.comment().as_string();
      anchor_rankt rank;
      if (mc == "termination per-loop marker")
        rank = is_lib ? LIB_LOOP : USER_LOOP;
      else if (mc == "termination abort-call marker")
        rank = is_lib ? LIB_ABORT : USER_ABORT;
      else
        continue;
      if (rank < best)
      {
        marker = it;
        best = rank;
      }
    }
    if (best == USER_LOOP)
      break;
  }
  if (best == NO_ANCHOR)
    return;

  goto_tracet trace;
  goto_trace_stept step;
  step.step_nr = 1;
  step.thread_nr = 0;
  step.type = goto_trace_stept::ASSERT;
  step.guard = false; // a violated assert
  step.pc = marker;
  step.comment = comment;
  trace.steps.push_back(step);

  if (want_graphml)
    violation_graphml_goto_trace(options, ns, trace);
  if (want_json)
    generate_json_report("1", ns, trace);
  if (want_sarif)
    sarif_goto_trace(options, ns, trace);
}

// This method iteratively applies one of the verification strategies
// for different unwinding bounds up to the specified maximum depth.
//
// ESBMC features 4 verification strategies:
//
//  1) Incremental
//  2) Termination
//  3) Falsification
//  4) k-induction
//
// Applying a strategy in this context means solving a particular sequence
// of decision problems from the list below for the given unwinding bound k:
//
//  - Base case             (see "is_base_case_violated")
//  - Forward condition     (see "does_forward_condition_hold")
//  - Inductive step        (see "is_inductive_step_violated")
//
// \param options - options for setting the verification strategy
// and controlling symbolic execution
// \param goto_functions - GOTO program under verification
int esbmc_parseoptionst::do_bmc_strategy(
  optionst &options,
  goto_functionst &goto_functions)
{
  // Get max number of iterations
  uint64_t max_k_step = cmdline.isset("unlimited-k-steps")
                          ? UINT_MAX
                          : strtoul(cmdline.getval("max-k-step"), nullptr, 10);

  // Get the increment
  unsigned k_step_inc = strtoul(cmdline.getval("k-step"), nullptr, 10);

  // Get the start of the base-case, default 1
  unsigned k_step_base = strtoul(cmdline.getval("base-k-step"), nullptr, 10);

  // For pytest test generation
  pytest_generator pytest_gen;

  // For ctest test generation
  ctest_generator ctest_gen;

  if (k_step_base >= max_k_step)
  {
    log_error(
      "Please specify --base-k-step smaller than max-k-step if you want "
      "to use incremental verification.");
    abort();
  }

  // Track whether any violation was found across all k steps.
  // In multi-property mode the loop continues past a violation to check
  // remaining properties, so we must remember the failure for the final verdict.
  bool any_violation_found = false;

  // Helper: emit the final verdict and return the correct exit code once a
  // proof or refutation has been found.  In multi-property mode the loop may
  // have continued past an earlier violation, so we must return 1 even when
  // the closing step (FC/IS) itself succeeds.
  auto conclude = [&]() -> int {
    // In coverage mode violations are expected; always report success.
    if (any_violation_found && !is_coverage)
    {
      log_fail("\nVERIFICATION FAILED");
      return 1;
    }
    return 0;
  };

  // Trying all bounds from 1 to "max_k_step" in "k_step_inc"
  uint64_t last_k_step = k_step_base;
  for (uint64_t k_step = k_step_base; k_step <= max_k_step;
       k_step += k_step_inc)
  {
    last_k_step = k_step;
    // k-induction
    if (options.get_bool_option("k-induction"))
    {
      bool is_bcv =
        is_base_case_violated(options, goto_functions, k_step).is_true();
      if (is_bcv)
      {
        any_violation_found = true;
        // Suppress spurious VERIFICATION SUCCESSFUL from report_result at
        // subsequent k steps where no new violations are found.
        options.set_option("kind-violation-found", true);
      }

      if (
        is_bcv && !cmdline.isset("multi-property") &&
        !options.get_bool_option("multi-property"))
        return 1;

      // if the property is proven violated in the bs, it's unnecessary to further run fw and is
      // this will make the trace looks cleaner yet might lead to an extra round to terminate the verification
      if (
        !is_bcv &&
        does_forward_condition_hold(options, goto_functions, k_step).is_false())
      {
        if (is_coverage)
          report_coverage(
            options,
            goto_functions.reached_claims,
            goto_functions.reached_mul_claims,
            pytest_gen,
            ctest_gen);
        return conclude();
      }

      // Don't run inductive step for k_step == 1
      if (k_step > 1)
      {
        if (
          !is_bcv && is_inductive_step_violated(options, goto_functions, k_step)
                       .is_false())
        {
          if (is_coverage)
            report_coverage(
              options,
              goto_functions.reached_claims,
              goto_functions.reached_mul_claims,
              pytest_gen,
              ctest_gen);
          return conclude();
        }
      }
    }
    // termination
    if (options.get_bool_option("termination"))
    {
      // `assert(false)` was inserted after main() and every loop havoc'd
      // by goto_termination. Property: "all executions terminate".
      //
      //   - Forward condition UNSAT at k:
      //       All states up to depth k are reachable — loops have fully
      //       unwound within k iters. Universal termination proven.
      //       Property HOLDS → return 0.
      //
      //   - Inductive step UNSAT at k:
      //       From no havoc'd iterate can the program reach end-of-main
      //       within k iters. A non-terminating execution exists.
      //       Property REFUTED → return 1.
      //
      // IS SAT is NOT a success condition: it only witnesses one
      // terminating path from one havoc'd state, which doesn't prove
      // all paths terminate.
      //
      // IS UNSAT is only sound when the k-induction havoc actually
      // covered the loop variables. Under --add-symex-value-sets,
      // loops that only modify pointers are SKIPPED by the havoc
      // transform (see goto_k_induction.cpp:91-94), so the IS just
      // runs the concrete initial state forward. IS UNSAT then means
      // "loop hasn't exited within k iters from initial state" —
      // which says nothing about non-termination; the loop may simply
      // need more iters. Disable the IS non-termination signal in
      // that mode and rely on FC alone.
      //
      // A linear ranking function proved every loop terminates (checked
      // once, before the havoc transform). This is k-independent, so
      // report success immediately without unwinding.
      if (options.get_bool_option("termination-ranking-proved"))
      {
        log_success(
          "\nRanking function shows all executions terminate\n"
          "VERIFICATION SUCCESSFUL");
        return 0;
      }

      // A recurrent-set non-termination check found an inductive R such
      // that every reachable state under R has an input continuation
      // staying in R and avoiding all exits (Gupta et al., POPL 2008).
      // The loop is non-terminating; report FAILED without unwinding.
      if (options.get_bool_option("termination-non-termination-proved"))
      {
        const std::string comment =
          "Recurrent set shows a non-terminating execution";
        const namespacet ns(context);
        report_non_termination_cwe(options, ns, goto_functions, comment);
        const std::string cwes = format_cwe_list(cwe_for(comment));
        std::string msg = "\n" + comment + "\n";
        if (!cwes.empty())
          msg += "CWE: " + cwes + "\n";
        msg += "VERIFICATION FAILED";
        log_fail("{}", msg);
        return 0;
      }

      // Skip IS for k = 1 (degenerates to a base-case check).
      if (does_forward_condition_hold(options, goto_functions, k_step)
            .is_false())
      {
        log_result(
          "\nForward condition shows all executions terminate "
          "(k = {:d})",
          k_step);
        return 0;
      }

      // IS UNSAT is only sound when k-induction actually havoc'd every
      // loop the property depends on. goto_k_induction skips a loop
      // when its modified set is empty — in that case
      // disable-inductive-step gets set mid-symex by the function-
      // pointer / recursion / concurrency hooks and the IS verdict is
      // treated as inconclusive below. Pointer-modifying loops are
      // now sound under --add-symex-value-sets thanks to the
      // value-set assume in symex_dereference, so no extra structural
      // gate is needed.
      if (k_step > 1)
      {
        tvt is_res =
          is_inductive_step_violated(options, goto_functions, k_step);
        // Symex may have set disable-inductive-step mid-run (function
        // pointers, recursion, concurrency). The IS UNSAT result is
        // then a vacuous "0 VCCs to falsify" and not a real
        // non-termination witness. Treat it as inconclusive.
        if (
          is_res.is_false() &&
          !options.get_bool_option("disable-inductive-step"))
        {
          const std::string comment = fmt::format(
            "Inductive step shows a non-terminating execution (k = {})",
            k_step);
          const namespacet ns(context);
          report_non_termination_cwe(options, ns, goto_functions, comment);
          const std::string cwes = format_cwe_list(cwe_for(comment));
          std::string msg = "\n" + comment;
          if (!cwes.empty())
            msg += "\nCWE: " + cwes;
          log_result("{}", msg);
          return 1;
        }
        // IS SAT or UNKNOWN — inconclusive, try larger k.
      }
    }
    // incremental-bmc
    if (options.get_bool_option("incremental-bmc"))
    {
      bool is_bcv =
        is_base_case_violated(options, goto_functions, k_step).is_true();
      if (is_bcv)
      {
        any_violation_found = true;
        options.set_option("kind-violation-found", true);
      }

      if (
        is_bcv && !cmdline.isset("multi-property") &&
        !options.get_bool_option("multi-property"))
        return 1;

      if (
        !is_bcv &&
        does_forward_condition_hold(options, goto_functions, k_step).is_false())
      {
        if (is_coverage)
          report_coverage(
            options,
            goto_functions.reached_claims,
            goto_functions.reached_mul_claims,
            pytest_gen,
            ctest_gen);
        return conclude();
      }
    }
    // falsification
    if (options.get_bool_option("falsification"))
    {
      if (is_base_case_violated(options, goto_functions, k_step).is_true())
        return 1;
    }
  }

  if (
    options.get_bool_option("multi-property") &&
    options.get_bool_option("k-induction"))
    diagnose_unknown_properties(options, goto_functions, last_k_step);

  log_status("Unable to prove or falsify the program, giving up.");
  log_fail("VERIFICATION UNKNOWN");

  if (is_coverage)
    report_coverage(
      options,
      goto_functions.reached_claims,
      goto_functions.reached_mul_claims,
      pytest_gen,
      ctest_gen);
  return 0;
}

// This checks whether "there is a set of inputs that reaches and violates
// an assertion when all the loops in the verified program are unwound up to
// the given bound k".
//
// \param options - options for controlling the symbolic execution
// \param goto_function - GOTO program under investigation
// \param k_step - depth to which all loops in the program are unrolled
// \return
//    TV_TRUE if such assertion violation (i.e., a bug) is found,
//    TV_FALSE if all reachable assertions hold for all input values
// in "goto_functions" with all its loops unrolled up to "k_step",
//    TV_UNKNOWN - otherwise.
tvt esbmc_parseoptionst::is_base_case_violated(
  optionst &options,
  goto_functionst &goto_functions,
  const uint64_t &k_step)
{
  options.set_option("base-case", true);
  options.set_option("forward-condition", false);
  options.set_option("inductive-step", false);
  options.set_option("no-unwinding-assertions", true);
  options.set_option("partial-loops", false);
  options.set_option("unwind", integer2string(k_step));

  bmct bmc(goto_functions, options, context);

  log_progress("Checking base case, k = {:d}", k_step);
  switch (do_bmc(bmc))
  {
  case P_UNSATISFIABLE:
    return tvt(tvt::TV_FALSE);

  case P_SMTLIB:
  case P_ERROR:
    break;

  case P_SATISFIABLE:
    log_result("\nBug found (k = {:d})", k_step);
    return tvt(tvt::TV_TRUE);

  default:
    log_result("Unknown BMC result");
    abort();
  }

  return tvt(tvt::TV_UNKNOWN);
}

// This checks whether "there is a set of inputs for which one of the loop
// conditions is still satisfied after it has been executed
// (i.e., unrolled) at least k times".
//
// \param options - options for controlling the symbolic execution
// \param goto_function - GOTO program under investigation
// \param k_step - depth to which all loops in the program are unrolled
// \return
//    TV_TRUE if there is a set of input values for which at least
// one of the loops in the program can be executed more than "k_step" times.
//    TV_FALSE if all reachable loops have at most "k_step" iterations
// for all input values in "goto_functions".
//    TV_UNKNOWN - otherwise.
tvt esbmc_parseoptionst::does_forward_condition_hold(
  optionst &options,
  goto_functionst &goto_functions,
  const uint64_t &k_step)
{
  if (options.get_bool_option("disable-forward-condition"))
    return tvt(tvt::TV_UNKNOWN);

  options.set_option("base-case", false);
  options.set_option("forward-condition", true);
  options.set_option("inductive-step", false);
  options.set_option("no-unwinding-assertions", false);
  options.set_option("partial-loops", false);

  // We have to disable assertions in the forward condition but
  // restore the previous value after it
  bool no_assertions = options.get_bool_option("no-assertions");

  // Turn assertions off
  options.set_option("no-assertions", true);
  options.set_option("unwind", integer2string(k_step));

  bmct bmc(goto_functions, options, context);

  log_progress("Checking forward condition, k = {:d}", k_step);
  auto res = do_bmc(bmc);

  // Restore the no assertion flag, before checking the other steps
  options.set_option("no-assertions", no_assertions);

  switch (res)
  {
  case P_SATISFIABLE:
    return tvt(tvt::TV_TRUE);

  case P_SMTLIB:
  case P_ERROR:
    break;

  case P_UNSATISFIABLE:
    log_result(
      "\nSolution found by the forward condition; "
      "all states are reachable (k = {:d})",
      k_step);
    return tvt(tvt::TV_FALSE);

  default:
    log_fail("Unknown BMC result");
    abort();
  }

  return tvt(tvt::TV_UNKNOWN);
}

// This tries to prove the inductive step: "assuming nondeterministic
// inputs for every loop, and assuming that all assertions hold for
// the first k iterations of every loop, all assertions will also hold
// when all loops in the program are unrolled to k+1."
// ("Loop inputs" are the variables whose values change inside the loop.)
//
// \param options - options for controlling the symbolic execution
// \param goto_function - GOTO program under investigation
// \param k_step - depth to which all loops in the program are unrolled
// \return -
//    TV_TRUE if there is a set of values for which all assertions in
// all loops hold for the first "k" iterations but not one of the assertions in
// one of the loops is violated during the "k+1" iterations.
//    TV_FALSE if the the inductive step holds.
//    TV_UNKNOWN - otherwise.
tvt esbmc_parseoptionst::is_inductive_step_violated(
  optionst &options,
  goto_functionst &goto_functions,
  const uint64_t &k_step)
{
  if (options.get_bool_option("disable-inductive-step"))
    return tvt(tvt::TV_UNKNOWN);

  if (strtoul(cmdline.getval("max-inductive-step"), nullptr, 10) < k_step)
    return tvt(tvt::TV_UNKNOWN);

  options.set_option("base-case", false);
  options.set_option("forward-condition", false);
  options.set_option("inductive-step", true);
  options.set_option("no-unwinding-assertions", true);
  options.set_option("partial-loops", true);
  options.set_option("unwind", integer2string(k_step));

  bmct bmc(goto_functions, options, context);

  log_progress("Checking inductive step, k = {:d}", k_step);
  smt_resultt res = static_cast<smt_resultt>(do_bmc(bmc));

  // Symex may flip `disable-inductive-step` mid-run when it encounters
  // a construct the IS cannot soundly handle (recursion, threads,
  // function-pointer calls). In that case the BMC result is the
  // outcome of an incomplete IS encoding — its UNSAT does not prove
  // safety. Discard the result and report UNKNOWN.
  if (options.get_bool_option("disable-inductive-step"))
    return tvt(tvt::TV_UNKNOWN);

  switch (res)
  {
  case P_SATISFIABLE:
    return tvt(tvt::TV_TRUE);

  case P_SMTLIB:
  case P_ERROR:
    break;

  case P_UNSATISFIABLE:
    log_result(
      "\nSolution found by the inductive step "
      "(k = {:d})",
      k_step);
    return tvt(tvt::TV_FALSE);

  default:
    log_fail("Unknown BMC result\n");
    abort();
  }

  return tvt(tvt::TV_UNKNOWN);
}

// This is a wrapper method that does a single round of
// symbolic execution of the given GOTO program and creates
// a decision problem specified by the verification options.
// In brief, they are used to control what assertions and
// assumptions are injected into the verified bounded trace
// during symbolic execution.
//
// \param bmc - the bmc object that contains all the necessary
// information (see below) to perform a single run of Bounded Model Checking:
//
//  1) GOTO program,
//  2) verification options.
//  3) program context,
int esbmc_parseoptionst::do_bmc(bmct &bmc)
{
  log_progress("Starting Bounded Model Checking");

  // Forward dead-store advisories (CWE-563) to bmc so they reach SARIF on both
  // the success and failure paths. Empty unless --dead-store-check is set.
  bmc.dead_store_advisories = dead_store_advisories;

  smt_resultt res;
  try
  {
    res = bmc.start_bmc();
  }
  catch (const inductive_step_disabled_exceptiont &e)
  {
    // Symex hit an IS-unsound construct (recursion, threads,
    // function-pointer call) and threw to short-circuit. Return
    // P_ERROR so the strategy layer drops to TV_UNKNOWN; the caller
    // also checks `disable-inductive-step` to suppress any verdict.
    log_status("Inductive step aborted: {}", e.reason);
    res = P_ERROR;
  }
  catch (const smtlib_convt::external_process_died &e)
  {
    // An external SMT solver process (an --smtlib solver, or the bitwuzllob
    // model solver) died or returned an unusable response at a point past the
    // backend's own recovery — e.g. while a counterexample was being read out
    // via (get-value). Report a clean failure rather than let the exception
    // reach std::terminate.
    log_error("SMT solver process failed: {}", e.what());
    res = P_ERROR;
  }

#ifdef HAVE_SENDFILE_ESBMC
  if (bmc.options.get_bool_option("memstats"))
  {
    int fd = open("/proc/self/status", O_RDONLY);
    sendfile(2, fd, nullptr, 100000);
    close(fd);
  }
#endif

  return res;
}

bool esbmc_parseoptionst::set_claims(goto_functionst &goto_functions)
{
  try
  {
    if (cmdline.isset("claim"))
      ::set_claims(goto_functions, cmdline.get_values("claim"));
  }

  catch (const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  catch (int)
  {
    return true;
  }

  return false;
}

// This method performs a wide range of actions that can be broadly divided
// into 3 main steps:
//
//  1) creating a GOTO program,
//  2) processing the GOTO program, and
//  3) outputting the GOTO program.
//
// This method is typically used as the second stage
// (right after parsing the command line options) by the verification methods
// (i.e., BMC, k-induction, etc).
//
// \param options - various options used during the above steps,
// \param goto_functions - the "created and processed" GOTO program.
bool esbmc_parseoptionst::get_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  try
  {
    fine_timet create_start = current_time();
    if (create_goto_program(options, goto_functions))
      return true;
    fine_timet create_stop = current_time();
    log_status(
      "GOTO program creation time: {}s",
      time2string(create_stop - create_start));

    fine_timet process_start = current_time();
    if (process_goto_program(options, goto_functions))
      return true;
    fine_timet process_stop = current_time();
    log_status(
      "GOTO program processing time: {}s",
      time2string(process_stop - process_start));
    if (output_goto_program(options, goto_functions))
      return true;
  }

  catch (const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  catch (std::bad_alloc &)
  {
    log_error("Out of memory");
    return true;
  }

  return false;
}

// Retarget the synthesised __ESBMC_main wrapper so its boilerplate call to
// c:@F@main instead dispatches to `target`. Used to bridge the entry point of a
// loaded --binary onto the program's real entry: a CBMC goto-binary's
// __CPROVER__start, or a user-selected --function harness. Without this,
// __ESBMC_main would run the empty boilerplate main and report a verdict over
// essentially no program. No-op if __ESBMC_main was not synthesised.
static void
retarget_esbmc_main(goto_functionst &goto_functions, const irep_idt &target)
{
  auto entry = goto_functions.function_map.find("__ESBMC_main");
  if (entry == goto_functions.function_map.end())
    return;

  Forall_goto_program_instructions (it, entry->second.body)
  {
    if (!it->is_function_call())
      continue;

    code_function_call2t &call = to_code_function_call2t(it->code);
    if (
      is_symbol2t(call.function) &&
      to_symbol2t(call.function).thename == "c:@F@main")
      call.function = symbol2tc(get_empty_type(), target);
  }
}

// This method creates a GOTO program from the source specified by the
// command line options. A GOTO program can be created:
//
//  1) from a GOTO binary file,
//  2) by parsing the input program files.
//
// \param options - options to be passed through,
// \param goto_functions - this is where the created GOTO program is stored.
static void link_cbmc_libm_bodies(goto_functionst &goto_functions);

bool esbmc_parseoptionst::create_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  try
  {
    if (cmdline.args.size() == 0)
    {
      log_error("Please provide a program to verify");
      return true;
    }

    // If the user is providing the GOTO functions, we don't need to parse
    if (cmdline.isset("binary"))
    {
      if (cmdline.isset("cprover"))
        log_warning(
          "--cprover is deprecated and has no effect; ESBMC additions are now "
          "linked automatically for CBMC goto-binaries (disable with "
          "--no-cprover-additions)");

      // A CBMC goto-binary needs ESBMC's additions (the __ESBMC_main entry
      // wrapper and the CPROVER-intrinsic bodies). Synthesise and link them
      // automatically, before reading the binaries so that goto_convert only
      // ever runs over the boilerplate and never clobbers the loaded bodies.
      const bool cbmc_additions =
        has_cbmc_binary_input() && !cmdline.isset("no-cprover-additions");
      if (cbmc_additions)
      {
        log_status(
          "CBMC goto-binary detected: linking ESBMC additions automatically");
        if (synthesize_cprover_additions(options, goto_functions))
          return true;
      }

      if (read_goto_binary(goto_functions))
        return true;

      // Resolve CBMC's bodyless libm externals (ceil/floor/trunc/round, ...) to
      // the operational-model bodies the additions linked, before symex sees a
      // bodyless call returning nondet.
      if (cbmc_additions)
        link_cbmc_libm_bodies(goto_functions);

      // Bridge the synthesised __ESBMC_main, which wraps the boilerplate
      // c:@F@main, onto the program's real entry. An explicit --function wins;
      // otherwise a CBMC binary dispatches into __CPROVER__start (it runs
      // __CPROVER_initialize and calls the program's main/harness). Without
      // this, a CBMC binary verifies the empty boilerplate main and may report
      // a spurious SUCCESSFUL.
      if (cmdline.isset("function"))
        retarget_esbmc_main(goto_functions, cmdline.getval("function"));
      else if (
        cbmc_additions && goto_functions.function_map.count("__CPROVER__start"))
        retarget_esbmc_main(goto_functions, "__CPROVER__start");
      else if (cbmc_additions)
        log_warning(
          "CBMC goto-binary support is experimental: no entry point to bridge "
          "(no __CPROVER__start and no --function), so __ESBMC_main wraps the "
          "boilerplate main and the verdict may be unsound.");

      goto_functions.update();
    }
    else
    {
      if (parse_goto_program(options, goto_functions))
        return true;
    }
  }

  catch (const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  catch (std::bad_alloc &)
  {
    log_error("Out of memory");
    return true;
  }

  return false;
}

// This method creates a GOTO program from the given GOTO binary.
//
// \param goto_functions - this is where the created GOTO program is stored.
bool esbmc_parseoptionst::read_goto_binary(goto_functionst &goto_functions)
{
  log_progress("Reading GOTO program from file");
  goto_binary_reader goto_reader;
  for (const auto &arg : cmdline.args)
  {
    if (goto_reader.read_goto_binary(arg, context, goto_functions))
    {
      log_error("Failed to open `{}'", arg);
      return true;
    }
  }

  return false;
}

// True if any --binary input is a CBMC goto-binary. CBMC's format starts with
// the magic 0x7f 'G' 'B' 'F'; ESBMC's own format starts with 'G' 'B' 'F'.
bool esbmc_parseoptionst::has_cbmc_binary_input()
{
  for (const auto &arg : cmdline.args)
  {
    std::ifstream in(arg, std::ios::in | std::ios::binary);
    unsigned char hdr[4] = {0, 0, 0, 0};
    // Bounded read: capped at sizeof(hdr) into the fixed-size buffer, and the
    // gcount() check confirms a full header before any byte is inspected
    // (CWE-120/CWE-20). Flawfinder: ignore
    in.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
    if (
      in.gcount() >= static_cast<std::streamsize>(sizeof(hdr)) &&
      is_cbmc_goto_magic(hdr))
      return true;
  }
  return false;
}

// Bridge CBMC's plain-named bodyless libm externals (e.g. `ceil`) to the
// operational-model bodies the additions link under the C-frontend-mangled id
// (`c:@F@ceil`). CBMC emits ceil/floor/trunc/round (and float/long-double
// variants) as bodyless FUNCTION_CALL externals; unlike sqrt/fabs -- rewritten
// to expressions in cbmc_adapter -- they have no ESBMC expression form and must
// run the library body. Copying the bodied function's body and type onto the
// bodyless declaration lets symex resolve the call: argument_assignments binds
// actual args using the copied type's parameter names, which match the copied
// body (goto-symex/symex_function.cpp).
static void link_cbmc_libm_bodies(goto_functionst &goto_functions)
{
  static const char *const libm[] = {
    "ceilf",     "ceil",     "ceill",     "floorf", "floor", "floorl",
    "truncf",    "trunc",    "truncl",    "roundf", "round", "roundl",
    "copysignf", "copysign", "copysignl", "fminf",  "fmin",  "fminl",
    "fmaxf",     "fmax",     "fmaxl",     "fdimf",  "fdim",  "fdiml"};

  for (const char *name : libm)
  {
    auto bodyless = goto_functions.function_map.find(name);
    if (
      bodyless == goto_functions.function_map.end() ||
      bodyless->second.body_available)
      continue;

    auto bodied = goto_functions.function_map.find(std::string("c:@F@") + name);
    if (
      bodied == goto_functions.function_map.end() ||
      !bodied->second.body_available)
      continue;

    bodyless->second.body = bodied->second.body; // operator= fixes up targets
    bodyless->second.type = bodied->second.type;
    bodyless->second.body_available = true;
  }
}

// Compile a boilerplate translation unit through the normal C-frontend pipeline
// to obtain ESBMC's "additions": typecheck() pulls in the C library via
// add_cprover_library, final() builds the __ESBMC_main entry wrapper, and
// goto_convert() turns them into goto bodies. This is exactly the prebuilt
// library.goto that one otherwise links manually. We run it before reading the
// CBMC binary so goto_convert only ever sees the boilerplate's symbols.
bool esbmc_parseoptionst::synthesize_cprover_additions(
  optionst &options,
  goto_functionst &goto_functions)
{
  file_operations::tmp_file tf =
    file_operations::create_tmp_file("esbmc-cprover-%%%%-%%%%-%%%%.c");
  // Taking the addresses of the bodied libm functions marks them referenced, so
  // add_cprover_library links their operational-model bodies into the additions;
  // link_cbmc_libm_bodies then bridges the CBMC binary's plain-named bodyless
  // declarations to them. Unlike sqrt/fabs (operators rewritten in cbmc_adapter)
  // these have no ESBMC expression form and must run the C library body.
  static const char boilerplate[] =
    "/* Auto-generated: bundle all ESBMC additions for CBMC gotos. */\n"
    "#include <math.h>\n"
    "void *const __esbmc_cbmc_libm_refs[] = {\n"
    "  (void *)ceilf,     (void *)ceil,     (void *)ceill,\n"
    "  (void *)floorf,    (void *)floor,    (void *)floorl,\n"
    "  (void *)truncf,    (void *)trunc,    (void *)truncl,\n"
    "  (void *)roundf,    (void *)round,    (void *)roundl,\n"
    "  (void *)copysignf, (void *)copysign, (void *)copysignl,\n"
    "  (void *)fminf,     (void *)fmin,     (void *)fminl,\n"
    "  (void *)fmaxf,     (void *)fmax,     (void *)fmaxl,\n"
    "  (void *)fdimf,     (void *)fdim,     (void *)fdiml,\n"
    "};\n"
    "int main(void) { return 0; }\n";
  if (fputs(boilerplate, tf.file()) == EOF || fflush(tf.file()) != 0)
  {
    log_error("could not write boilerplate for CPROVER additions");
    return true;
  }

  // Point the command line at the boilerplate for this one compile, then
  // restore it for the subsequent binary read. The user's --function names a
  // harness in the CBMC binary, not in this boilerplate, so also neutralise
  // config.main (which config.cpp sets from --function) for the compile --
  // otherwise the boilerplate's entry-point synthesis (clang_c_main) looks for
  // that harness here and aborts with "main symbol not found". The retarget in
  // create_goto_program applies --function to the loaded binary afterwards.
  cmdlinet::argst saved_args = cmdline.args;
  const std::string saved_main = config.main;
  cmdline.args = {tf.path()};
  config.main = "";
  bool failed = parse_goto_program(options, goto_functions);
  cmdline.args = saved_args;
  config.main = saved_main;

  if (failed)
    log_error("failed to synthesize ESBMC additions for CBMC goto-binary");
  return failed;
}

// This method creates a GOTO program by parsing the input program files.
//
// \param options - options to be passed to the program parser,
// \param goto_functions - this is where the created GOTO program is stored.
bool esbmc_parseoptionst::parse_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  try
  {
    if (parse(cmdline))
      return true;

    if (cmdline.isset("parse-tree-too") || cmdline.isset("parse-tree-only"))
    {
      std::ostringstream oss;
      for (auto &it : langmap)
        it.second->show_parse(oss);
      log_status("{}", oss.str());
      if (cmdline.isset("parse-tree-only"))
        exit(0);
    }

    // Typechecking (old frontend) or adjust (clang frontend)
    if (typecheck())
      return true;
    if (final())
      return true;

    // we no longer need any parse trees or language files
    clear_parse();

    if (cmdline.isset("symbol-table-too") || cmdline.isset("symbol-table-only"))
    {
      std::ostringstream oss;
      show_symbol_table_plain(oss);
      log_status("{}", oss.str());
      if (cmdline.isset("symbol-table-only"))
        exit(0);
    }

    // Expand --no-standard-checks into individual options before goto_convert,
    // because VLA size checks are generated during goto conversion.
    if (
      cmdline.isset("no-standard-checks") ||
      options.get_bool_option("no-standard-checks"))
    {
      options.set_option("no-pointer-check", true);
      options.set_option("no-div-by-zero-check", true);
      options.set_option("no-pointer-relation-check", true);
      options.set_option("no-unlimited-scanf-check", true);
      options.set_option("no-vla-size-check", true);
      options.set_option("no-align-check", true);
      options.set_option("no-bounds-check", true);
    }

    log_progress("Generating GOTO Program");
    goto_convert(context, options, goto_functions);
  }

  catch (const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  catch (std::bad_alloc &)
  {
    log_error("Out of memory");
    return true;
  }

  return false;
}

// This method provides different output methods for the given GOTO program.
// Depending on the provided options this method can:
//
//  - output the given GOTO program as text,
//  - translate the provided GOTO program into C,
//  - create a GOTO binary from this GOTO program,
//  - methods outputting some additional information of the GOTO program.
//
// \param options - various options setting the output methods,
// \param goto_functions - the GOTO program to be output.
bool esbmc_parseoptionst::output_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  try
  {
    namespacet ns(context);

    // show it?
    if (cmdline.isset("show-loops"))
    {
      show_loop_numbers(goto_functions);
      return true;
    }

    // show it?
    if (cmdline.isset("show-goto-value-sets"))
    {
      value_set_analysist value_set_analysis(ns);
      value_set_analysis(goto_functions);
      std::ostringstream oss;
      show_value_sets(goto_functions, value_set_analysis, oss);
      log_result("{}", oss.str());
      return true;
    }

    // Write the GOTO program into a binary
    if (cmdline.isset("output-goto"))
    {
      log_status("Writing GOTO program to file");
      std::ofstream oss(
        cmdline.getval("output-goto"), std::ios::out | std::ios::binary);
      if (write_goto_binary(oss, context, goto_functions))
      {
        log_error("Failed to generate goto binary file"); // TODO: explain why
        abort();
      };
      return true;
    }

    if (cmdline.isset("show-ileave-points"))
    {
      print_ileave_points(ns, goto_functions);
      return true;
    }

    // Output the GOTO program to the log (and terminate or continue) in
    // a human-readable format
    if (
      cmdline.isset("goto-functions-too") ||
      cmdline.isset("goto-functions-only"))
    {
      std::ostringstream oss;
      goto_functions.output(ns, oss);
      log_status("{}", oss.str());
      if (cmdline.isset("goto-functions-only"))
        exit(0);
    }

    if (cmdline.isset("dump-goto-cfg"))
    {
      goto_cfg cfg(goto_functions);
      cfg.dump_graph();
      return true;
    }

    // Print a flat list of every function call site with its arguments.
    // Output format: caller -> callee(arg1, arg2, ...) [file:line]
    // Nested calls appear as separate lines with compiler-generated
    // temporaries (e.g. return_value$_add$5) showing data flow.
    if (cmdline.isset("show-call-sites"))
    {
      for (const auto &f : goto_functions.function_map)
      {
        if (!f.second.body_available)
          continue;
        const std::string caller = f.first.as_string();
        forall_goto_program_instructions (i_it, f.second.body)
        {
          if (i_it->is_function_call())
          {
            const auto &fc = to_code_function_call2t(i_it->code);

            // Direct calls have a symbol; indirect calls (function
            // pointers) fall back to pretty-printing the expression.
            std::string callee;
            if (is_symbol2t(fc.function))
              callee = to_symbol2t(fc.function).get_symbol_name();
            else
              callee = from_expr(ns, "", fc.function);

            // Pretty-print each actual argument as a comma-separated list
            std::string args;
            for (size_t i = 0; i < fc.operands.size(); i++)
            {
              if (i > 0)
                args += ", ";
              args += from_expr(ns, "", fc.operands[i]);
            }

            std::string loc;
            if (!i_it->location.get_file().empty())
            {
              const auto &file = i_it->location.get_file();
              const auto &line = i_it->location.get_line();

              if (!line.empty())
                loc = " [" + file.as_string() + ":" + line.as_string() + "]";
              else
                loc = " [" + file.as_string() + "]";
            }
            log_status("{} -> {}({}){}", caller, callee, args, loc);
          }
        }
      }
      std::exit(0);
    }

    // Translate the GOTO program to C and output it into the log or
    // a specified output file
    if (cmdline.isset("goto2c"))
    {
      // Creating a translator here
      goto2ct goto2c(ns, goto_functions);
      goto2c.preprocess();
      goto2c.check();
      std::string res = goto2c.translate();

      const std::string &filename = options.get_option("output");
      if (!filename.empty())
      {
        // Outputting the translated program into the output file
        std::ofstream out(filename);
        assert(out);
        out << res;
      }
      else
        std::cout << res;
      return true;
    }
  }

  catch (const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  return false;
}

// This performs the preprocessing of the input program
// when the old C/C++ frontend (i.e., from "ansi-c/" or "cpp/") is used.
void esbmc_parseoptionst::preprocessing()
{
  try
  {
    if (cmdline.args.size() != 1)
    {
      log_error("Please provide one program to preprocess");
      return;
    }

    std::string filename = cmdline.args[0];

    // To test that the file exists,
    std::ifstream infile(filename.c_str());
    if (!infile)
    {
      log_error("failed to open input file");
      return;
    }
#ifdef ENABLE_OLD_FRONTEND
    std::ostringstream oss;
    if (c_preprocess(filename, oss, false))
      log_error("PREPROCESSING ERROR");
    log_status("{}", oss.str());
#endif
  }
  catch (const char *e)
  {
    log_error("{}", e);
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
  }

  catch (std::bad_alloc &)
  {
    log_error("Out of memory");
  }
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
      (std::isalnum(text[i + 2]) || text[i + 2] == '-'))
    {
      size_t start = i;
      i += 2;
      while (i < text.size() && (std::isalnum(text[i]) || text[i] == '-'))
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
  auto const inner =
    80 - title_start.length() - title_end.length() - esbmc_string.length();
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

// When k-induction exhausts all k-steps without a definitive result, run one
// final per-VCC inductive-step check at the last k to identify which specific
// properties could not be resolved, without impacting the main k-induction loop.
void esbmc_parseoptionst::diagnose_unknown_properties(
  optionst &options,
  goto_functionst &goto_functions,
  const uint64_t k_step)
{
  if (options.get_bool_option("disable-inductive-step"))
    return;

  // Mirror the guards used by is_inductive_step_violated in the main loop:
  // inductive step is skipped for k==1 and capped by --max-inductive-step.
  if (k_step <= 1)
    return;
  if (strtoul(cmdline.getval("max-inductive-step"), nullptr, 10) < k_step)
    return;

  const bool saved_base_case = options.get_bool_option("base-case");
  const bool saved_forward_condition =
    options.get_bool_option("forward-condition");
  const bool saved_inductive_step = options.get_bool_option("inductive-step");
  const bool saved_no_unwinding =
    options.get_bool_option("no-unwinding-assertions");
  const bool saved_partial_loops = options.get_bool_option("partial-loops");
  const std::string saved_unwind = options.get_option("unwind");

  options.set_option("base-case", false);
  options.set_option("forward-condition", false);
  options.set_option("inductive-step", true);
  options.set_option("no-unwinding-assertions", true);
  options.set_option("partial-loops", true);
  options.set_option("unwind", integer2string(k_step));
  options.set_option("diagnose-unknown-properties", true);

  bmct bmc(goto_functions, options, context);

  log_progress(
    "\nDiagnosing unresolved properties (inductive step, k = {:d}):", k_step);
  do_bmc(bmc);

  options.set_option("base-case", saved_base_case);
  options.set_option("forward-condition", saved_forward_condition);
  options.set_option("inductive-step", saved_inductive_step);
  options.set_option("no-unwinding-assertions", saved_no_unwinding);
  options.set_option("partial-loops", saved_partial_loops);
  options.set_option("unwind", saved_unwind);
  options.set_option("diagnose-unknown-properties", false);
}
