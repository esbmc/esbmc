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
#include <esbmc/globals.h>
#include <goto-symex/engine/goto_symex.h>
#include <goto-symex/trace/goto_trace.h>
#include <goto-symex/trace/sarif.h>
#include <util/base/cwe_mapping.h>
#include <solvers/smt/smt_result.h>
#include <solvers/smtlib/smtlib_conv.h>
#include <solvers/solve.h>
#include <cctype>
#include <charconv>
#include <clang-c-frontend/clang_c_language.h>
#include <util/config/config.h>
#include <util/base/filesystem.h>
#include <util/base/signal_catcher.h>
#include <csignal>
#include <cstdlib>
#include <limits>
#include <util/expr/expr_util.h>
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
#include <goto-programs/mark_decl_as_non_det.h>
#include <goto-programs/assign_params_as_non_det.h>
#include <goto2c/goto2c.h>
#include <util/irep/irep.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <memory>
#include <pointer-analysis/goto_program_dereference.h>
#include <pointer-analysis/show_value_sets.h>
#include <pointer-analysis/value_set_analysis.h>
#include <util/symtab/symbol.h>
#include <util/base/time_stopping.h>
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

#ifndef _WIN32
// "Data even on UNKNOWN": a run killed mid-solve never reaches
// report_coverage, so the handlers below emit the partial coverage summary
// first. Async-signal-safe: atomic loads, a stack buffer and write(2) only.
namespace
{
struct sigbuf
{
  char buf[512];
  size_t n = 0;
  void put(const char *s)
  {
    while (*s && n < sizeof(buf))
      buf[n++] = *s++;
  }
  void put_uint(size_t v)
  {
    char tmp[24];
    size_t t = 0;
    do
    {
      tmp[t++] = static_cast<char>('0' + v % 10);
      v /= 10;
    } while (v && t < sizeof(tmp));
    while (t && n < sizeof(buf))
      buf[n++] = tmp[--t];
  }
  void flush()
  {
    signal_safe_write(STDOUT_FILENO, buf, n);
    n = 0;
  }
};

// Path coverage counts complete paths, not decision edges, so it gets its own
// summary rather than borrowing the branch metric's heading. The number is a
// lower bound and carries no counterexample payload; that lives in
// cov-ce-journal.json when --cov-report-json was given.
void emit_path_coverage_on_signal()
{
  if (!goto_coveraget::path_cov_active.load(std::memory_order_relaxed))
    return;
  const size_t total =
    goto_coveraget::total_paths_atomic.load(std::memory_order_relaxed);
  const size_t decided =
    goto_coveraget::live_decided.load(std::memory_order_relaxed);
  const size_t claims =
    goto_coveraget::claims_total_atomic.load(std::memory_order_relaxed);
  size_t f = goto_coveraget::live_F.load(std::memory_order_relaxed);
  if (f > total && total) // the numerator can never exceed the universe
    f = total;

  sigbuf b;
  b.put(
    "\n[Coverage]\nReport Completeness: PARTIAL — terminated by signal "
    "before verification concluded\nComplete Paths : ");
  b.put_uint(total);
  b.put("\nClaims Decided : ");
  b.put_uint(decided);
  b.put(" of ");
  b.put_uint(claims);
  b.put("\nPath Status: F ");
  b.put_uint(f);
  b.put(
    " (partial: LOWER BOUND, no cov-report.json was written, and this line "
    "carries no counterexample payload. The payload for these paths is in "
    "cov-ce-journal.json when --cov-report-json was given)\n");
  b.flush();
}

void emit_coverage_on_signal()
{
  emit_path_coverage_on_signal();
  if (!goto_coveraget::branch_cov_active.load(std::memory_order_relaxed))
    return;
  const size_t total =
    goto_coveraget::total_branch_atomic.load(std::memory_order_relaxed);
  if (total == 0)
    return;
  // Covered-set runs count |all_claims ∩ (covered_set ∪ reached)|; default
  // runs count reached_claims. Both are re-synced at every report_coverage,
  // so this is exact after the first report and a lower bound before it.
  size_t reached =
    goto_coveraget::covered_set_mode.load(std::memory_order_relaxed)
      ? goto_coveraget::covered_run.load(std::memory_order_relaxed)
      : goto_coveraget::live_reached.load(std::memory_order_relaxed);
  if (reached > total) // the numerator can never exceed the universe
    reached = total;

  // Same shape as report_coverage's branch block, so stdout consumers parse
  // a killed run and a concluded one identically.
  sigbuf b;
  b.put("\n[Coverage]\nBranches : ");
  b.put_uint(total);
  b.put("\nReached : ");
  b.put_uint(reached);
  b.put("\nBranch Coverage: ");
  b.put_uint(reached * 100 / total);
  b.put("% (partial: run terminated before verification concluded)\n");
  b.flush();
}
} // namespace

// External bounders (the regression harness, timeout(1), CI runners) kill
// with SIGTERM or SIGINT rather than ESBMC's own SIGALRM. Emit the partial
// coverage in the grace window before SIGKILL, then shut down as usual.
void term_handler(int sig)
{
  emit_coverage_on_signal();
  signal_catcher(sig);
}

// Runs on SIGALRM, on whatever the main thread was doing — quite possibly
// inside malloc holding the arena lock, which is why every call here is from
// POSIX's async-signal-safe set (#6201). has_violation() is an atomic load.
void timeout_handler(int)
{
  static const char timed_out[] = "ERROR: Timed out\n";
  static const char failed[] = "VERIFICATION FAILED\n";

  emit_coverage_on_signal();
  signal_safe_write(STDERR_FILENO, timed_out, sizeof(timed_out) - 1);
  // Under --multi-property the run keeps exploring interleavings after a
  // violation, to reach the properties only later schedules touch. A timeout
  // landing in that tail must not discard a counterexample already found and
  // printed: the verdict is settled once a property is violated.
  if (goto_functionst::property_verdicts.has_violation())
    signal_safe_write(STDERR_FILENO, failed, sizeof(failed) - 1);
  // Kill any external solver process groups first: they are in their own
  // groups, so they outlive this _exit() otherwise (e.g. an mpirun job).
  file_operations::kill_registered_pgroups_from_signal();
  file_operations::remove_registered_tmps_from_signal();
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
  // Shared by the __GLIBC__ backtrace block and the /proc/self/maps dump
  // below; cppcheck suggests reducing its scope only in the non-glibc
  // configuration, where the backtrace use is compiled out.
  // cppcheck-suppress variableScope
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
    for (ssize_t rd;
         (rd = read(fd, buffer, sizeof(buffer))) > 0 || // Flawfinder: ignore
         (rd == -1 && errno == EINTR);)
      rd = write(STDERR_FILENO, buffer, rd < 0 ? 0 : rd);
    close(fd);
  }
  ::raise(sig);
}

/* Displaces the concise reporter installed before doit(): asking for the
 * backtrace is explicit intent, so this one does not defer to a handler
 * somebody else set. */
static void install_verbose_crash_handler(const cmdlinet &cmdline)
{
  if (!cmdline.isset("segfault-handler"))
    return;

  for (int sig : {SIGSEGV, SIGBUS, SIGABRT})
    install_altstack_handler(sig, segfault_handler, false);
}
#else
static void install_verbose_crash_handler(const cmdlinet &)
{
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
  size_t digits = str.size();
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
    --digits;
  }
  uint64_t timeout = 0;
  auto [ptr, ec] = std::from_chars(str.data(), str.data() + digits, timeout);
  if (ec != std::errc() || ptr != str.data() + digits)
  {
    log_error("Invalid timeout value '{}'", std::string(str));
    abort();
  }
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
  size_t digits = str.size();
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
    --digits;
  }
  uint64_t size = 0;
  auto [ptr, ec] = std::from_chars(str.data(), str.data() + digits, size);
  if (ec != std::errc() || ptr != str.data() + digits)
  {
    log_error("Invalid memlimit value '{}'", std::string(str));
    abort();
  }
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

/// Option defaults the loop-invariant modes imply.
///
/// --synthesise-loop-invariants: the schema emits establishment, preservation
/// and the post-loop use as three independent obligations. Bundling them into
/// one query makes the solver carry every multiplier term at once -- on
/// regression/esbmc/synth_loop_invariant_sum each discharges in about a second
/// alone while the bundle does not finish in 120s -- so solve them separately.
/// base-case is set alongside multi-property because multi_property_check only
/// runs when both are on.
///
/// Vacuity: default-enable the probe under --loop-invariant-check (the
/// standalone Hoare-rewrite mode). A loop invariant that implies the guard
/// makes the post-loop continuation unreachable; without this probe every
/// downstream claim discharges as vacuously true. Deliberately NOT
/// default-enabled for combined mode --loop-invariant: that runs k-induction
/// phases whose UNSAT-on-internal-claims is the success signal, not vacuity.
/// Users can opt in explicitly with --check-vacuity there.
static void
set_loop_invariant_options(const cmdlinet &cmdline, optionst &options)
{
  // Only when no phase has been selected explicitly. --base-case,
  // --forward-condition and --inductive-step each drive one k-induction phase
  // themselves, and process_goto_program treats --inductive-step as
  // k-induction, so it stamps inductive_step_instruction on the havoc. Forcing
  // base-case on top of that reaches the (base_case || forward_condition) &&
  // inductive_step_instruction arm with k_induction false, which symex asserts
  // against (execution_state.cpp) -- ESBMC aborts.
  if (
    cmdline.isset("synthesise-loop-invariants") && !options.is_kind() &&
    !cmdline.isset("base-case") && !cmdline.isset("forward-condition") &&
    !cmdline.isset("inductive-step"))
  {
    options.set_option("multi-property", true);
    options.set_option("base-case", true);
  }

  if (cmdline.isset("no-vacuity-check"))
    options.set_option("check-vacuity", false);
  else if (
    cmdline.isset("check-vacuity") || cmdline.isset("loop-invariant-check") ||
    cmdline.isset("synthesise-loop-invariants"))
    options.set_option("check-vacuity", true);
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
    log_result("{}", esbmc_build_id());
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

  if (cmdline.isset("incremental-context-bound"))
    options.set_option("incremental-context-bound", true);

  if (cmdline.isset("max-context-bound"))
    options.set_option(
      "max-context-bound", cmdline.getval("max-context-bound"));
  else
    options.set_option("max-context-bound", 20);

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

  // A base-case start of k = 0 sets --unwind 0, which ESBMC treats as
  // "unlimited" (loop termination is gated on max_unwind != 0), so the base
  // case would unwind forever instead of running k = 0. Reject it up front,
  // independently of --unlimited-k-steps (discussion #6093).
  if (strtoul(cmdline.getval("base-k-step"), nullptr, 10) == 0)
  {
    log_error(
      "Please specify --base-k-step >= 1: a base case of k = 0 sets "
      "--unwind 0, which ESBMC treats as unlimited unwinding.");
    abort();
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

    if (k_step_inc >= max_k_step)
    {
      log_error(
        "--k-step ({}) must be smaller than --max-k-step ({}).",
        k_step_inc,
        max_k_step);
      abort();
    }

    if (k_step_base >= max_k_step)
    {
      log_error(
        "--base-k-step ({}) must be smaller than --max-k-step ({}).",
        k_step_base,
        max_k_step);
      abort();
    }
  }

  if (cmdline.isset("coverage-multi-tx"))
  {
    // --coverage-multi-tx keeps the multi-transaction dispatcher loop LIVE in
    // Solidity coverage mode (see the neutralization site below), and the
    // Foundry generator reconstructs the ordered call sequence from that loop's
    // per-iteration dispatcher-guard markers.
    //
    // --solidity-max-tx N instead makes get_tx_bound() emit a DETERMINISTIC
    // unroll of N transaction bodies (no loop). That reconstructs unreliably —
    // the generator can mis-attribute which method ran in each transaction (it
    // emitted `fire(); fire();` where `arm(); fire();` was needed) — so the two
    // flags are incompatible: route the transaction bound through the live loop.
    if (cmdline.isset("solidity-max-tx"))
    {
      log_error(
        "--coverage-multi-tx is incompatible with --solidity-max-tx: an "
        "explicit tx bound forces a deterministic unroll whose Foundry "
        "reconstruction is unreliable. Bound the live dispatcher loop with "
        "--incremental-bmc (recommended) or --unwind N instead.");
      abort();
    }
    // The live loop is an unbounded `while(nondet_bool())`; symex needs a
    // whole-program bounding strategy or it diverges. Loop-specific bounds
    // (--unwindset / --unwindsetname) are NOT accepted here: they target a named
    // loop, not necessarily the dispatcher, so they cannot guarantee the loop is
    // bounded. Require an explicit global strategy rather than guessing.
    if (
      !cmdline.isset("unwind") && !cmdline.isset("incremental-bmc") &&
      !cmdline.isset("k-induction") && !cmdline.isset("k-induction-parallel") &&
      !cmdline.isset("termination") && !cmdline.isset("base-case") &&
      !cmdline.isset("forward-condition") && !cmdline.isset("inductive-step"))
    {
      log_error(
        "--coverage-multi-tx keeps the unbounded multi-transaction dispatcher "
        "loop live; it needs a global bounding strategy. Add --incremental-bmc "
        "(recommended: discovers the transaction depth dynamically) or "
        "--unwind N (N reaches up to ~N-1 transactions).");
      abort();
    }
  }

  // The deterministic-unroll path (--solidity-max-tx N, N>=2) reconstructs a
  // multi-transaction Foundry sequence UNRELIABLY: the generator can
  // mis-attribute which method ran in each transaction (emitting fire();fire();
  // where arm();fire(); was needed). Surface it and point at the reliable path.
  if (
    cmdline.isset("generate-foundry-testcase") &&
    cmdline.isset("solidity-max-tx") && !cmdline.isset("coverage-multi-tx"))
  {
    const long n = strtol(cmdline.getval("solidity-max-tx"), nullptr, 10);
    if (n >= 2)
      log_warning(
        "--solidity-max-tx {} with --generate-foundry-testcase reconstructs "
        "multi-transaction sequences unreliably (methods can be mis-attributed "
        "across transactions). For reliable ordered sequences use "
        "--coverage-multi-tx --incremental-bmc instead.",
        n);
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

  set_loop_invariant_options(cmdline, options);

  // Conflicting strategies: --termination checks a different property and
  // takes priority over k-induction. Disable both k-induction variants so
  // only the termination strategy runs (#6031).
  if (options.is_kind() && cmdline.isset("termination"))
  {
    log_warning(
      "Both k-induction and --termination specified. Using --termination.");
    options.set_option("k-induction", false);
    options.set_option("k-induction-parallel", false);
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

#ifndef _WIN32
  // Unconditional (independent of --timeout, which external harnesses
  // strip): emit partial branch coverage on external SIGTERM/SIGINT
  // before the killer's SIGKILL. See term_handler.
  signal(SIGTERM, term_handler);
  signal(SIGINT, term_handler);
#endif

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

  install_verbose_crash_handler(cmdline);

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

  if (cmdline.isset("ts-check"))
    options.set_option("k-induction", true);

  if (cmdline.isset("dump-violation-info"))
    options.set_option(
      "dump-violation-info", cmdline.getval("dump-violation-info"));

  // Solidity complete-path coverage needs REVERTS to be observable rather than
  // path-pruned: a `require`/`revert` failure is one of the paths it enumerates,
  // and the legacy lowering both erases it (`__ESBMC_assume(false)`) and leaves
  // the surviving edge byte-identical to a plain early `return`, so the exit
  // cannot be classified. Publish the flag as a boolean here — BEFORE
  // `config.options` is captured — because the frontend runs long before the
  // coverage dispatch, and `get_bool_option` on the raw NULL-valued CLI flag
  // would read as false. Setting it enables the in-tree revert-observation gate
  // (solidity_convert.cpp), which is regression-locked by the assert_revert_*
  // suite and, unlike --bound, does not change external-call modelling.
  if (cmdline.isset("solidity-path-coverage"))
    options.set_option("solidity-path-coverage-enabled", true);

  // ---- --path-cov-arith-resolve: REFUSE rather than silently do nothing ----
  //
  // The mechanism re-solves a witnessed path claim with goto_check's own
  // arithmetic conditions assumed. Those conditions only EXIST if a check was
  // enabled, so without one there is nothing to assume and the flag would be a
  // no-op that looks like a fix. This tool has already shipped a function that
  // was never called and a guard that was always true; a third mechanism whose
  // absence is invisible is not acceptable.
  //
  // The dependency is stated rather than papered over. Setting the check
  // options from here would have to survive whatever the Solidity
  // standard-checks expansion does to them afterwards, and a mechanism that
  // depends on option-application ORDER is one that breaks the day the order
  // changes -- silently, in the direction of doing nothing.
  if (cmdline.isset("path-cov-arith-resolve"))
  {
    if (!cmdline.isset("solidity-path-coverage"))
    {
      log_error("--path-cov-arith-resolve is only meaningful with "
                "--solidity-path-coverage: it re-solves a COMPLETE PATH claim, "
                "and no other mode emits one.");
      abort();
    }
    if (
      !cmdline.isset("overflow-check") && !cmdline.isset("div-by-zero-check") &&
      !cmdline.isset("unsigned-overflow-check"))
    {
      log_error(
        "--path-cov-arith-resolve needs the conditions it is supposed to "
        "assume, and no arithmetic check is enabled. Add --overflow-check "
        "and/or --div-by-zero-check. Refusing rather than running: with no "
        "check enabled goto_check emits no overflow / division-by-zero claim, "
        "so this flag would re-solve nothing and report a clean run -- "
        "indistinguishable from the defect being fixed.");
      abort();
    }
    options.set_option("path-cov-arith-resolve", true);
    log_status(
      "--path-cov-arith-resolve: a witnessed path whose counterexample "
      "violates an enabled arithmetic check will be re-solved ONCE with that "
      "check's own condition assumed. A non-wrapping witness replaces the "
      "wrapping one; if none exists the path is reachable only through a "
      "checked-arithmetic revert, which is counted in its own cell and NOT "
      "folded into U. The cost -- claims re-solved and seconds spent -- is "
      "printed at the end of the run rather than left to be inferred.");
  }

  config.options = options;
}
