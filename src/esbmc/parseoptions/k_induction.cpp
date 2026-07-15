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
  // Both members are read/written through whole-struct pipe I/O below and
  // consumed at a_result.type / a_result.k; cppcheck sees them as unused
  // only in the _WIN32 configuration, where the parallel k-induction body
  // is compiled out.
  // cppcheck-suppress unusedStructMember
  PROCESS_TYPE type;
  // cppcheck-suppress unusedStructMember
  uint64_t k;
};

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
    log_error("Child processes were not created successfully.");
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
                          ? std::numeric_limits<uint64_t>::max()
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
      int read_size = read( // Flawfinder: ignore
        forward_pipe[0],
        &a_result,
        sizeof(resultt));
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
      // and haven't crashed (if it crashed, solution will be max_k_step)
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
      // and haven't crashed (if it crashed, solution will be max_k_step)
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
      int read_size = read( // Flawfinder: ignore
        backward_pipe[0],
        &a_result,
        sizeof(resultt));
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
