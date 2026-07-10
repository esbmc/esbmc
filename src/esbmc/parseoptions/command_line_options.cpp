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

extern "C" const char buildidstring_buf[];
extern "C" const unsigned int buildidstring_buf_size;

static std::string_view esbmc_version_string()
{
  return {buildidstring_buf, buildidstring_buf_size};
}

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
