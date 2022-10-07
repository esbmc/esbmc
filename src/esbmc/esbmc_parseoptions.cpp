#include <ac_config.h>

#ifndef _WIN32
extern "C"
{
#include <fcntl.h>
#include <unistd.h>

#ifdef HAVE_SENDFILE_ESBMC
#include <sys/sendfile.h>
#endif

#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
}
#endif

#include <esbmc/bmc.h>
#include <esbmc/esbmc_parseoptions.h>
#include <cctype>
#include <clang-c-frontend/clang_c_language.h>
#include <util/config.h>
#include <csignal>
#include <cstdlib>
#include <util/expr_util.h>
#include <fstream>
#include <goto-programs/add_race_assertions.h>
#include <goto-programs/goto_check.h>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_inline.h>
#include <goto-programs/goto_k_induction.h>
#include <goto-programs/interval_analysis.h>
#include <goto-programs/loop_numbers.h>
#include <goto-programs/read_goto_binary.h>
#include <goto-programs/write_goto_binary.h>
#include <goto-programs/remove_skip.h>
#include <goto-programs/remove_unreachable.h>
#include <goto-programs/set_claims.h>
#include <goto-programs/show_claims.h>
#include <goto-programs/loop_unroll.h>
#include <goto-programs/mark_decl_as_non_det.h>
#include <util/irep.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <memory>
#include <pointer-analysis/goto_program_dereference.h>
#include <pointer-analysis/show_value_sets.h>
#include <pointer-analysis/value_set_analysis.h>
#include <util/symbol.h>
#include <util/time_stopping.h>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#ifdef ENABLE_OLD_FRONTEND
#include <ansi-c/c_preprocess.h>
#endif

#ifdef ENABLE_GOTO_CONTRACTOR
#include <goto-programs/goto_contractor.h>
#endif

enum PROCESS_TYPE
{
  BASE_CASE,
  FORWARD_CONDITION,
  INDUCTIVE_STEP,
  PARENT
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
  // Unfortunately some highly useful pieces of code hook themselves into
  // aexit and attempt to free some memory. That doesn't really make sense to
  // occur on exit, but more importantly doesn't mix well with signal handlers,
  // and results in the allocator locking against itself. So use _exit instead
  _exit(1);
}
#endif

void esbmc_parseoptionst::set_verbosity_msg()
{
  VerbosityLevel v = VerbosityLevel::Status;

  if(cmdline.isset("verbosity"))
  {
    v = (VerbosityLevel)atoi(cmdline.getval("verbosity"));
    if(v < VerbosityLevel::None)
      v = VerbosityLevel::None;
    else if(v > VerbosityLevel::Debug)
      v = VerbosityLevel::Debug;
  }

  messaget::state.verbosity = v;
}

extern "C" const uint8_t *const esbmc_version_string;

uint64_t esbmc_parseoptionst::read_time_spec(const char *str)
{
  uint64_t mult;
  int len = strlen(str);
  if(!isdigit(str[len - 1]))
  {
    switch(str[len - 1])
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
  else
  {
    mult = 1;
  }

  uint64_t timeout = strtol(str, nullptr, 10);
  timeout *= mult;
  return timeout;
}

uint64_t esbmc_parseoptionst::read_mem_spec(const char *str)
{
  uint64_t mult;
  int len = strlen(str);
  if(!isdigit(str[len - 1]))
  {
    switch(str[len - 1])
    {
    case 'b':
      mult = 1;
      break;
    case 'k':
      mult = 1024;
      break;
    case 'm':
      mult = 1024 * 1024;
      break;
    case 'g':
      mult = 1024 * 1024 * 1024;
      break;
    default:
      log_error("Unrecognized memlimit suffix");
      abort();
    }
  }
  else
  {
    mult = 1024 * 1024;
  }

  uint64_t size = strtol(str, nullptr, 10);
  size *= mult;
  return size;
}

static std::string format_target()
{
  const char *endian = nullptr;
  switch(config.ansi_c.endianess)
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
  switch(config.ansi_c.lib)
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

void esbmc_parseoptionst::get_command_line_options(optionst &options)
{
  if(config.set(cmdline))
  {
    exit(1);
  }
  log_status("Target: {}", format_target());

  options.cmdline(cmdline);
  set_verbosity_msg();

  if(cmdline.isset("cex-output"))
    options.set_option("cex-output", cmdline.getval("cex-output"));

  /* graphML generation options check */
  if(cmdline.isset("witness-output"))
    options.set_option("witness-output", cmdline.getval("witness-output"));

  if(cmdline.isset("witness-producer"))
    options.set_option("witness-producer", cmdline.getval("witness-producer"));

  if(cmdline.isset("witness-programfile"))
    options.set_option(
      "witness-programfile", cmdline.getval("witness-programfile"));

  if(cmdline.isset("git-hash"))
  {
    log_result("{}", esbmc_version_string);
    exit(0);
  }

  if(cmdline.isset("list-solvers"))
  {
    // Generated for us by autoconf,
    log_result("Available solvers: {}", ESBMC_AVAILABLE_SOLVERS);
    exit(0);
  }

  if(cmdline.isset("bv"))
  {
    options.set_option("int-encoding", false);
  }

  if(cmdline.isset("ir"))
  {
    options.set_option("int-encoding", true);
  }

  if(cmdline.isset("fixedbv"))
    options.set_option("fixedbv", true);
  else
    options.set_option("floatbv", true);

  if(cmdline.isset("context-bound"))
    options.set_option("context-bound", cmdline.getval("context-bound"));
  else
    options.set_option("context-bound", -1);

  if(cmdline.isset("lock-order-check"))
    options.set_option("lock-order-check", true);

  if(cmdline.isset("deadlock-check"))
  {
    options.set_option("deadlock-check", true);
    options.set_option("atomicity-check", false);
  }
  else
    options.set_option("deadlock-check", false);

  if(cmdline.isset("compact-trace"))
    options.set_option("no-slice", true);

  if(cmdline.isset("smt-during-symex"))
  {
    log_status("Enabling --no-slice due to presence of --smt-during-symex");
    options.set_option("no-slice", true);
  }

  if(cmdline.isset("smt-thread-guard") || cmdline.isset("smt-symex-guard"))
  {
    if(!cmdline.isset("smt-during-symex"))
    {
      log_error(
        "Please explicitly specify --smt-during-symex if you want "
        "to use features that involve encoding SMT during symex");
      abort();
    }
  }

  // check the user's parameters to run incremental verification
  if(!cmdline.isset("unlimited-k-steps"))
  {
    // Get max number of iterations
    BigInt max_k_step = strtoul(cmdline.getval("max-k-step"), nullptr, 10);

    // Get the increment
    unsigned k_step_inc = strtoul(cmdline.getval("k-step"), nullptr, 10);

    // check whether k-step is greater than max-k-step
    if(k_step_inc >= max_k_step)
    {
      log_error(
        "Please specify --k-step smaller than max-k-step if you want "
        "to use incremental verification.");
      abort();
    }
  }

  if(cmdline.isset("base-case"))
  {
    options.set_option("base-case", true);
    options.set_option("no-unwinding-assertions", true);
    options.set_option("partial-loops", false);
  }

  if(cmdline.isset("forward-condition"))
  {
    options.set_option("forward-condition", true);
    options.set_option("no-unwinding-assertions", false);
    options.set_option("partial-loops", false);
    options.set_option("no-assertions", true);
  }

  if(cmdline.isset("inductive-step"))
  {
    options.set_option("inductive-step", true);
    options.set_option("no-unwinding-assertions", true);
    options.set_option("partial-loops", false);
  }

  if(cmdline.isset("overflow-check"))
  {
    options.set_option("disable-inductive-step", true);
  }

  if(cmdline.isset("timeout"))
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

  if(cmdline.isset("memlimit"))
  {
#ifdef _WIN32
    log_error("Can't memlimit on Windows, sorry");
    abort();
#else
    uint64_t size = read_mem_spec(cmdline.getval("memlimit"));

    struct rlimit lim;
    lim.rlim_cur = size;
    lim.rlim_max = size;
    if(setrlimit(RLIMIT_DATA, &lim) != 0)
    {
      perror("Couldn't set memory limit");
      abort();
    }
#endif
  }

#ifndef _WIN32
  struct rlimit lim;
  if(cmdline.isset("enable-core-dump"))
  {
    lim.rlim_cur = RLIM_INFINITY;
    lim.rlim_max = RLIM_INFINITY;
    if(setrlimit(RLIMIT_CORE, &lim) != 0)
    {
      perror("Couldn't unlimit core dump size");
      abort();
    }
  }
  else
  {
    lim.rlim_cur = 0;
    lim.rlim_max = 0;
    if(setrlimit(RLIMIT_CORE, &lim) != 0)
    {
      perror("Couldn't disable core dump size");
      abort();
    }
  }
#endif

  // If multi-property is on, we should set result-only
  if(cmdline.isset("multi-property"))
    options.set_option("result-only", true);

  config.options = options;
}

int esbmc_parseoptionst::doit()
{
  // Configure msg output

  if(cmdline.isset("file-output"))
  {
    FILE *f = fopen(cmdline.getval("file-output"), "w+");
    out = f;
    err = f;
    messaget::state.err = f;
    messaget::state.out = f;
  }
  //
  // Print a banner
  //
  log_status(
    "ESBMC version {} {}-bit {} {}",
    ESBMC_VERSION,
    sizeof(void *) * 8,
    config.this_architecture(),
    config.this_operating_system());

  if(cmdline.isset("version"))
    return 0;

  // unwinding of transition systems

  if(cmdline.isset("module") || cmdline.isset("gen-interface"))
  {
    log_error(
      "This version has no support for "
      " hardware modules.");
    return 1;
  }

  // command line options
  if(cmdline.isset("preprocess"))
  {
    preprocessing();
    return 0;
  }

  // initialize goto_functions algorithms
  {
    // loop unroll
    if(cmdline.isset("goto-unwind") && !cmdline.isset("unwind"))
    {
      size_t unroll_limit = cmdline.isset("unlimited-goto-unwind") ? -1 : 1000;
      goto_preprocess_algorithms.push_back(
        std::make_unique<bounded_loop_unroller>(unroll_limit));
    }

    // mark declarations as nondet
    if(cmdline.isset("initialize-nondet-variables"))
      goto_preprocess_algorithms.emplace_back(
        std::make_unique<mark_decl_as_non_det>(context));
  }
  if(cmdline.isset("termination"))
    return doit_termination();

  if(cmdline.isset("incremental-bmc"))
    return doit_incremental();

  if(cmdline.isset("falsification"))
    return doit_falsification();

  if(cmdline.isset("k-induction"))
    return doit_k_induction();

  if(cmdline.isset("k-induction-parallel"))
    return doit_k_induction_parallel();

  optionst opts;
  get_command_line_options(opts);

  if(get_goto_program(opts, goto_functions))
    return 6;

  if(cmdline.isset("show-claims"))
  {
    const namespacet ns(context);
    show_claims(ns, goto_functions);
    return 0;
  }

  if(set_claims(goto_functions))
    return 7;

  if(opts.get_bool_option("skip-bmc"))
    return 0;

  // do actual BMC
  bmct bmc(goto_functions, opts, context);

  return do_bmc(bmc);
}

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

  if(pipe(forward_pipe))
  {
    log_status("\nPipe Creation Failed, giving up.");
    _exit(1);
  }

  if(pipe(backward_pipe))
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
  for(unsigned p = 0; p < 3; ++p)
  {
    pid_t pid = fork();

    if(pid == -1)
    {
      log_status("\nFork Failed, giving up.");
      _exit(1);
    }

    // Child process
    if(!pid)
    {
      process_type = PROCESS_TYPE(p);
      break;
    }
    // Parent process

    children_pid[p] = pid;
    ++num_p;
  }

  if(process_type == PARENT && num_p != 3)
  {
    log_error("Child processes were not created sucessfully.");
    abort();
  }

  optionst opts;

  if(process_type != PARENT)
  {
    // Get full set of options
    get_command_line_options(opts);

    // Generate goto functions and set claims
    if(get_goto_program(opts, goto_functions))
      return 6;

    if(cmdline.isset("show-claims"))
    {
      const namespacet ns(context);
      show_claims(ns, goto_functions);
      return 0;
    }

    if(set_claims(goto_functions))
      return 7;
  }

  // Get max number of iterations
  BigInt max_k_step = cmdline.isset("unlimited-k-steps")
                        ? UINT_MAX
                        : strtoul(cmdline.getval("max-k-step"), nullptr, 10);

  // Get the increment
  unsigned k_step_inc = strtoul(cmdline.getval("k-step"), nullptr, 10);

  // All processes were created successfully
  switch(process_type)
  {
  case PARENT:
  {
    // Communication to child processes
    close(forward_pipe[1]);
    close(backward_pipe[0]);

    struct resultt a_result;
    bool bc_finished = false, fc_finished = false, is_finished = false;
    BigInt bc_solution = max_k_step, fc_solution = max_k_step,
           is_solution = max_k_step;

    // Keep reading until we find an answer
    while(!(bc_finished && fc_finished && is_finished))
    {
      // Perform read and interpret the number of bytes read
      int read_size = read(forward_pipe[0], &a_result, sizeof(resultt));
      if(read_size != sizeof(resultt))
      {
        if(read_size == 0)
        {
          // Client hung up; continue on, but don't interpret the result.
          ;
        }
        else
        {
          // Invalid size read.
          log_error("Short read communicating with kinduction children");
          log_error("Size {}, expected {}", read_size, sizeof(resultt));
          abort();
        }
      }

      // Eventually the parent process will check if the child process is alive

      // Check base case process
      if(!bc_finished)
      {
        int status;
        pid_t result = waitpid(children_pid[0], &status, WNOHANG);
        if(result == 0)
        {
          // Child still alive
        }
        else if(result == -1)
        {
          // Error
        }
        else
        {
          log_warning("base case process crashed.");
          bc_finished = fc_finished = is_finished = true;
        }
      }

      // Check forward condition process
      if(!fc_finished)
      {
        int status;
        pid_t result = waitpid(children_pid[1], &status, WNOHANG);
        if(result == 0)
        {
          // Child still alive
        }
        else if(result == -1)
        {
          // Error
        }
        else
        {
          log_warning("forward condition process crashed.");
          fc_finished = bc_finished = is_finished = true;
        }
      }

      // Check inductive step process
      if(!is_finished)
      {
        int status;
        pid_t result = waitpid(children_pid[2], &status, WNOHANG);
        if(result == 0)
        {
          // Child still alive
        }
        else if(result == -1)
        {
          // Error
        }
        else
        {
          log_warning("inductive step process crashed.");
          is_finished = bc_finished = fc_finished = true;
        }
      }

      switch(a_result.type)
      {
      case BASE_CASE:
        bc_finished = true;
        bc_solution = a_result.k;
        break;

      case FORWARD_CONDITION:
        fc_finished = true;
        fc_solution = a_result.k;
        break;

      case INDUCTIVE_STEP:
        is_finished = true;
        is_solution = a_result.k;
        break;

      default:
        log_error(
          "Message from unrecognized k-induction child "
          "process");
        abort();
      }

      // If either the base case found a bug or the forward condition
      // finds a solution, present the result
      if(bc_finished && (bc_solution != 0) && (bc_solution != max_k_step))
        break;

      // If the either the forward condition or inductive step finds a
      // solution, first check if base case couldn't find a bug in that code,
      // if there is no bug, inductive step can present the result
      if(fc_finished && (fc_solution != 0) && (fc_solution != max_k_step))
      {
        // If base case finished, then we can present the result
        if(bc_finished)
          break;

        // Otherwise, kill the inductive step process
        kill(children_pid[2], SIGKILL);

        // And ask base case for a solution

        // Struct to keep the result
        struct resultt r = {process_type, 0};

        r.k = fc_solution.to_uint64();

        // Write result
        auto const len = write(backward_pipe[1], &r, sizeof(r));
        assert(len == sizeof(r) && "short write");
        (void)len; //ndebug
      }

      if(is_finished && (is_solution != 0) && (is_solution != max_k_step))
      {
        // If base case finished, then we can present the result
        if(bc_finished)
          break;

        // Otherwise, kill the forward condition process
        kill(children_pid[1], SIGKILL);

        // And ask base case for a solution

        // Struct to keep the result
        struct resultt r = {process_type, 0};

        r.k = is_solution.to_uint64();

        // Write result
        auto const len = write(backward_pipe[1], &r, sizeof(r));
        assert(len == sizeof(r) && "short write");
        (void)len; //ndebug
      }
    }

    for(int i : children_pid)
      kill(i, SIGKILL);

    // Check if a solution was found by the base case
    if(bc_finished && (bc_solution != 0) && (bc_solution != max_k_step))
    {
      log_result(
        "\nBug found by the base case (k = {})\nVERIFICATION FAILED",
        bc_solution);
      return true;
    }

    // Check if a solution was found by the forward condition
    if(fc_finished && (fc_solution != 0) && (fc_solution != max_k_step))
    {
      // We should only present the result if the base case finished
      // and haven't crashed (if it crashed, bc_solution will be UINT_MAX
      if(bc_finished && (bc_solution != max_k_step))
      {
        log_result(
          "\nSolution found by the forward condition; "
          "all states are reachable (k = {:d})\n"
          "VERIFICATION SUCCESSFUL",
          fc_solution);
        return false;
      }
    }

    // Check if a solution was found by the inductive step
    if(is_finished && (is_solution != 0) && (is_solution != max_k_step))
    {
      // We should only present the result if the base case finished
      // and haven't crashed (if it crashed, bc_solution will be UINT_MAX
      if(bc_finished && (bc_solution != max_k_step))
      {
        log_result(
          "\nSolution found by the inductive step "
          "(k = {:d})\n"
          "VERIFICATION SUCCESSFUL",
          is_solution);
        return false;
      }
    }

    // Couldn't find a bug or a proof for the current deepth
    log_result("\nVERIFICATION UNKNOWN");
    return false;
  }

  case BASE_CASE:
  {
    // Set that we are running base case
    opts.set_option("base-case", true);
    opts.set_option("forward-condition", false);
    opts.set_option("inductive-step", false);

    opts.set_option("no-unwinding-assertions", true);
    opts.set_option("partial-loops", false);

    // Start communication to the parent process
    close(forward_pipe[0]);
    close(backward_pipe[1]);

    // Struct to keep the result
    struct resultt r = {process_type, 0};

    // Run bmc and only send results in two occasions:
    // 1. A bug was found, we send the step where it was found
    // 2. It couldn't find a bug
    for(BigInt k_step = 1; k_step <= max_k_step; k_step += k_step_inc)
    {
      bmct bmc(goto_functions, opts, context);
      bmc.options.set_option("unwind", integer2string(k_step));

      log_status("Checking base case, k = {:d}\n", k_step);

      // If an exception was thrown, we should abort the process
      int res = smt_convt::P_ERROR;
      try
      {
        res = do_bmc(bmc);
      }
      catch(...)
      {
        break;
      }

      // Send information to parent if no bug was found
      if(res == smt_convt::P_SATISFIABLE)
      {
        r.k = k_step.to_uint64();

        // Write result
        auto const len = write(forward_pipe[1], &r, sizeof(r));
        assert(len == sizeof(r) && "short write");
        (void)len; //ndebug

        log_status("BASE CASE PROCESS FINISHED.\n");
        return true;
      }

      // Check if the parent process is asking questions

      // Perform read and interpret the number of bytes read
      struct resultt a_result;
      int read_size = read(backward_pipe[0], &a_result, sizeof(resultt));
      if(read_size != sizeof(resultt))
      {
        if(read_size == 0)
        {
          // Client hung up; continue on, but don't interpret the result.
          continue;
        }
        if(read_size == -1 && errno == EAGAIN)
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
      if(a_result.k < k_step)
        break;

      // Otherwise, we just need to check the base case for k = a_result.k
      max_k_step = a_result.k + k_step_inc;
    }

    // Send information to parent that a bug was not found
    r.k = 0;

    auto const len = write(forward_pipe[1], &r, sizeof(r));
    assert(len == sizeof(r) && "short write");
    (void)len; //ndebug

    log_status("BASE CASE PROCESS FINISHED.\n");
    return false;
  }

  case FORWARD_CONDITION:
  {
    // Set that we are running forward condition
    opts.set_option("base-case", false);
    opts.set_option("forward-condition", true);
    opts.set_option("inductive-step", false);

    opts.set_option("no-unwinding-assertions", false);
    opts.set_option("partial-loops", false);
    opts.set_option("no-assertions", true);

    // Start communication to the parent process
    close(forward_pipe[0]);
    close(backward_pipe[1]);

    // Struct to keep the result
    struct resultt r = {process_type, 0};

    // Run bmc and only send results in two occasions:
    // 1. A proof was found, we send the step where it was found
    // 2. It couldn't find a proof
    for(BigInt k_step = 2; k_step <= max_k_step; k_step += k_step_inc)
    {
      bmct bmc(goto_functions, opts, context);
      bmc.options.set_option("unwind", integer2string(k_step));

      log_status("Checking forward condition, k = {:d}", k_step);

      // If an exception was thrown, we should abort the process
      int res = smt_convt::P_ERROR;
      try
      {
        res = do_bmc(bmc);
      }
      catch(...)
      {
        break;
      }

      if(opts.get_bool_option("disable-forward-condition"))
        break;

      // Send information to parent if no bug was found
      if(res == smt_convt::P_UNSATISFIABLE)
      {
        r.k = k_step.to_uint64();

        // Write result
        auto const len = write(forward_pipe[1], &r, sizeof(r));
        assert(len == sizeof(r) && "short write");
        (void)len; //ndebug

        log_status("FORWARD CONDITION PROCESS FINISHED.");
        return false;
      }
    }

    // Send information to parent that it couldn't prove the code
    r.k = 0;

    auto const len = write(forward_pipe[1], &r, sizeof(r));
    assert(len == sizeof(r) && "short write");
    (void)len; //ndebug

    log_status("FORWARD CONDITION PROCESS FINISHED.");
    return true;
  }

  case INDUCTIVE_STEP:
  {
    // Set that we are running inductive step
    opts.set_option("base-case", false);
    opts.set_option("forward-condition", false);
    opts.set_option("inductive-step", true);

    opts.set_option("no-unwinding-assertions", true);
    opts.set_option("partial-loops", true);

    // Start communication to the parent process
    close(forward_pipe[0]);
    close(backward_pipe[1]);

    // Struct to keep the result
    struct resultt r = {process_type, 0};

    // Run bmc and only send results in two occasions:
    // 1. A proof was found, we send the step where it was found
    // 2. It couldn't find a proof
    for(BigInt k_step = 2; k_step <= max_k_step; k_step += k_step_inc)
    {
      bmct bmc(goto_functions, opts, context);

      bmc.options.set_option("unwind", integer2string(k_step));

      log_status("Checking inductive step, k = {:d}", k_step);

      // If an exception was thrown, we should abort the process
      int res = smt_convt::P_ERROR;
      try
      {
        res = do_bmc(bmc);
      }
      catch(...)
      {
        break;
      }

      if(opts.get_bool_option("disable-inductive-step"))
        break;

      // Send information to parent if no bug was found
      if(res == smt_convt::P_UNSATISFIABLE)
      {
        r.k = k_step.to_uint64();

        // Write result
        auto const len = write(forward_pipe[1], &r, sizeof(r));
        assert(len == sizeof(r) && "short write");
        (void)len; //ndebug

        log_status("INDUCTIVE STEP PROCESS FINISHED.");
        return false;
      }
    }

    // Send information to parent that it couldn't prove the code
    r.k = 0;

    auto const len = write(forward_pipe[1], &r, sizeof(r));
    assert(len == sizeof(r) && "short write");
    (void)len; //ndebug

    log_status("INDUCTIVE STEP PROCESS FINISHED.");
    return true;
  }

  default:
    assert(0 && "Unknown process type.");
  }

#endif

  return 0;
}

int esbmc_parseoptionst::doit_k_induction()
{
  optionst opts;
  get_command_line_options(opts);

  if(get_goto_program(opts, goto_functions))
    return 6;

  if(cmdline.isset("show-claims"))
  {
    const namespacet ns(context);
    show_claims(ns, goto_functions);
    return 0;
  }

  if(set_claims(goto_functions))
    return 7;

  // Get max number of iterations
  BigInt max_k_step = cmdline.isset("unlimited-k-steps")
                        ? UINT_MAX
                        : strtoul(cmdline.getval("max-k-step"), nullptr, 10);

  // Get the increment
  unsigned k_step_inc = strtoul(cmdline.getval("k-step"), nullptr, 10);

  for(BigInt k_step = 1; k_step <= max_k_step; k_step += k_step_inc)
  {
    if(do_base_case(opts, goto_functions, k_step))
      return true;

    if(!do_forward_condition(opts, goto_functions, k_step))
      return false;

    if(!do_inductive_step(opts, goto_functions, k_step))
      return false;
  }

  log_status("Unable to prove or falsify the program, giving up.");
  log_status("VERIFICATION UNKNOWN");

  return 0;
}

int esbmc_parseoptionst::doit_falsification()
{
  optionst opts;
  get_command_line_options(opts);

  if(get_goto_program(opts, goto_functions))
    return 6;

  if(cmdline.isset("show-claims"))
  {
    const namespacet ns(context);
    show_claims(ns, goto_functions);
    return 0;
  }

  if(set_claims(goto_functions))
    return 7;

  // Get max number of iterations
  BigInt max_k_step = cmdline.isset("unlimited-k-steps")
                        ? UINT_MAX
                        : strtoul(cmdline.getval("max-k-step"), nullptr, 10);

  // Get the increment
  unsigned k_step_inc = strtoul(cmdline.getval("k-step"), nullptr, 10);

  for(BigInt k_step = 1; k_step <= max_k_step; k_step += k_step_inc)
  {
    if(do_base_case(opts, goto_functions, k_step))
      return true;
  }

  log_status("Unable to prove or falsify the program, giving up.");
  log_status("VERIFICATION UNKNOWN");

  return 0;
}

int esbmc_parseoptionst::doit_incremental()
{
  optionst opts;
  get_command_line_options(opts);

  if(get_goto_program(opts, goto_functions))
    return 6;

  if(cmdline.isset("show-claims"))
  {
    const namespacet ns(context);
    std::ostringstream oss;
    show_claims(ns, goto_functions);
    return 0;
  }

  if(set_claims(goto_functions))
    return 7;

  // Get max number of iterations
  BigInt max_k_step = cmdline.isset("unlimited-k-steps")
                        ? UINT_MAX
                        : strtoul(cmdline.getval("max-k-step"), nullptr, 10);

  // Get the increment
  unsigned k_step_inc = strtoul(cmdline.getval("k-step"), nullptr, 10);

  for(BigInt k_step = 1; k_step <= max_k_step; k_step += k_step_inc)
  {
    if(do_base_case(opts, goto_functions, k_step))
      return true;

    if(!do_forward_condition(opts, goto_functions, k_step))
      return false;
  }

  log_status("Unable to prove or falsify the program, giving up.");
  log_status("VERIFICATION UNKNOWN");

  return 0;
}

int esbmc_parseoptionst::doit_termination()
{
  optionst opts;
  get_command_line_options(opts);

  if(get_goto_program(opts, goto_functions))
    return 6;

  if(cmdline.isset("show-claims"))
  {
    const namespacet ns(context);
    show_claims(ns, goto_functions);
    return 0;
  }

  if(set_claims(goto_functions))
    return 7;

  // Get max number of iterations
  BigInt max_k_step = cmdline.isset("unlimited-k-steps")
                        ? UINT_MAX
                        : strtoul(cmdline.getval("max-k-step"), nullptr, 10);

  // Get the increment
  unsigned k_step_inc = strtoul(cmdline.getval("k-step"), nullptr, 10);

  for(BigInt k_step = 1; k_step <= max_k_step; k_step += k_step_inc)
  {
    if(!do_forward_condition(opts, goto_functions, k_step))
      return false;

    /* Disable this for now as it is causing more than 100 errors on SV-COMP
    if(!do_inductive_step(opts, goto_functions, k_step))
      return false;
    */
  }

  log_status("Unable to prove or falsify the program, giving up.");
  log_status("VERIFICATION UNKNOWN");

  return 0;
}

int esbmc_parseoptionst::do_base_case(
  optionst &opts,
  goto_functionst &goto_functions,
  const BigInt &k_step)
{
  opts.set_option("base-case", true);
  opts.set_option("forward-condition", false);
  opts.set_option("inductive-step", false);

  opts.set_option("no-unwinding-assertions", true);
  opts.set_option("partial-loops", false);

  bmct bmc(goto_functions, opts, context);

  bmc.options.set_option("unwind", integer2string(k_step));

  log_status("Checking base case, k = {:d}", k_step);
  switch(do_bmc(bmc))
  {
  case smt_convt::P_UNSATISFIABLE:
  case smt_convt::P_SMTLIB:
  case smt_convt::P_ERROR:
    break;

  case smt_convt::P_SATISFIABLE:
    log_result("\nBug found (k = {:d})", k_step);
    return true;

  default:
    log_result("Unknown BMC result");
    abort();
  }

  return false;
}

int esbmc_parseoptionst::do_forward_condition(
  optionst &opts,
  goto_functionst &goto_functions,
  const BigInt &k_step)
{
  if(opts.get_bool_option("disable-forward-condition"))
    return true;

  opts.set_option("base-case", false);
  opts.set_option("forward-condition", true);
  opts.set_option("inductive-step", false);

  opts.set_option("no-unwinding-assertions", false);
  opts.set_option("partial-loops", false);

  // We have to disable assertions in the forward condition but
  // restore the previous value after it
  bool no_assertions = opts.get_bool_option("no-assertions");

  // Turn assertions off
  opts.set_option("no-assertions", true);

  bmct bmc(goto_functions, opts, context);

  bmc.options.set_option("unwind", integer2string(k_step));

  log_status("Checking forward condition, k = {:d}", k_step);
  auto res = do_bmc(bmc);

  // Restore the no assertion flag, before checking the other steps
  opts.set_option("no-assertions", no_assertions);

  switch(res)
  {
  case smt_convt::P_SATISFIABLE:
  case smt_convt::P_SMTLIB:
  case smt_convt::P_ERROR:
    break;

  case smt_convt::P_UNSATISFIABLE:
    log_result(
      "\nSolution found by the forward condition; "
      "all states are reachable (k = {:d})",
      k_step);
    return false;

  default:
    log_result("Unknown BMC result");
    abort();
  }

  return true;
}

int esbmc_parseoptionst::do_inductive_step(
  optionst &opts,
  goto_functionst &goto_functions,
  const BigInt &k_step)
{
  // Don't run inductive step for k_step == 1
  if(k_step == 1)
    return true;

  if(opts.get_bool_option("disable-inductive-step"))
    return true;

  if(
    strtoul(cmdline.getval("max-inductive-step"), nullptr, 10) <
    k_step.to_uint64())
    return true;

  opts.set_option("base-case", false);
  opts.set_option("forward-condition", false);
  opts.set_option("inductive-step", true);

  opts.set_option("no-unwinding-assertions", true);
  opts.set_option("partial-loops", true);

  bmct bmc(goto_functions, opts, context);
  bmc.options.set_option("unwind", integer2string(k_step));

  log_status("Checking inductive step, k = {:d}", k_step);
  switch(do_bmc(bmc))
  {
  case smt_convt::P_SATISFIABLE:
  case smt_convt::P_SMTLIB:
  case smt_convt::P_ERROR:
    break;

  case smt_convt::P_UNSATISFIABLE:
    log_result(
      "\nSolution found by the inductive step "
      "(k = {:d})",
      k_step);
    return false;

  default:
    log_result("Unknown BMC result\n");
    abort();
  }

  return true;
}

bool esbmc_parseoptionst::set_claims(goto_functionst &goto_functions)
{
  try
  {
    if(cmdline.isset("claim"))
      ::set_claims(goto_functions, cmdline.get_values("claim"));
  }

  catch(const char *e)
  {
    log_error(e);
    return true;
  }

  catch(const std::string &e)
  {
    log_error(e);
    return true;
  }

  catch(int)
  {
    return true;
  }

  return false;
}

bool esbmc_parseoptionst::get_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  fine_timet parse_start = current_time();
  try
  {
    if(cmdline.args.size() == 0)
    {
      log_error("Please provide a program to verify");
      return true;
    }

    // Ahem
    migrate_namespace_lookup = new namespacet(context);

    // If the user is providing the GOTO functions, we don't need to parse
    if(cmdline.isset("binary"))
    {
      log_status("Reading GOTO program from file");

      if(read_goto_binary(goto_functions))
        return true;
    }
    else
    {
      // Parsing
      if(parse())
        return true;

      if(cmdline.isset("parse-tree-too") || cmdline.isset("parse-tree-only"))
      {
        assert(language_files.filemap.size());
        languaget &language = *language_files.filemap.begin()->second.language;
        std::ostringstream oss;
        language.show_parse(oss);
        log_status("{}", oss.str());
        if(cmdline.isset("parse-tree-only"))
          return true;
      }

      // Typecheking (old frontend) or adjust (clang frontend)
      if(typecheck())
        return true;
      if(final())
        return true;

      // we no longer need any parse trees or language files
      clear_parse();

      if(
        cmdline.isset("symbol-table-too") || cmdline.isset("symbol-table-only"))
      {
        std::ostringstream oss;
        show_symbol_table_plain(oss);
        log_status("{}", oss.str());
        if(cmdline.isset("symbol-table-only"))
          return true;
      }

      log_status("Generating GOTO Program");

      goto_convert(context, options, goto_functions);
    }

    fine_timet parse_stop = current_time();
    log_status(
      "GOTO program creation time: {}s", time2string(parse_stop - parse_start));

    fine_timet process_start = current_time();
    if(process_goto_program(options, goto_functions))
      return true;
    fine_timet process_stop = current_time();
    log_status(
      "GOTO program processing time: {}s",
      time2string(process_stop - process_start));
  }

  catch(const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch(const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  catch(std::bad_alloc &)
  {
    log_error("Out of memory");
    return true;
  }

  return false;
}

void esbmc_parseoptionst::preprocessing()
{
  try
  {
    if(cmdline.args.size() != 1)
    {
      log_error("Please provide one program to preprocess");
      return;
    }

    std::string filename = cmdline.args[0];

    // To test that the file exists,
    std::ifstream infile(filename.c_str());
    if(!infile)
    {
      log_error("failed to open input file");
      return;
    }
#ifdef ENABLE_OLD_FRONTEND
    std::ostringstream oss;
    if(c_preprocess(filename, oss, false))
      log_error("PREPROCESSING ERROR");
    log_status("{}", oss.str());
#endif
  }
  catch(const char *e)
  {
    log_error("{}", e);
  }

  catch(const std::string &e)
  {
    log_error("{}", e);
  }

  catch(std::bad_alloc &)
  {
    log_error("Out of memory");
  }
}

bool esbmc_parseoptionst::read_goto_binary(goto_functionst &goto_functions)
{
  for(const auto &arg : _cmdline.args)
  {
    if(::read_goto_binary(arg, context, goto_functions))
    {
      log_error("Failed to open `{}'", arg);
      return true;
    }
  }

  return false;
}

bool esbmc_parseoptionst::process_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  try
  {
    namespacet ns(context);
    for(auto &algorithm : goto_preprocess_algorithms)
      algorithm->run(goto_functions);

    // do partial inlining
    if(!cmdline.isset("no-inlining"))
    {
      if(cmdline.isset("full-inlining"))
        goto_inline(goto_functions, options, ns);
      else
        goto_partial_inline(goto_functions, options, ns);
    }

    if(cmdline.isset("interval-analysis") || cmdline.isset("goto-contractor"))
    {
      interval_analysis(goto_functions, ns);
    }

    if(
      cmdline.isset("inductive-step") || cmdline.isset("k-induction") ||
      cmdline.isset("k-induction-parallel"))
    {
      goto_k_induction(goto_functions);
    }

    if(cmdline.isset("goto-contractor"))
    {
#ifdef ENABLE_GOTO_CONTRACTOR
      goto_contractor(goto_functions);
#else
      log_error(
        "Current build does not support contractors. If ibex is installed, add "
        "-DENABLE_IBEX = ON");
      abort();
#endif
    }

    if(cmdline.isset("termination"))
    {
      goto_termination(goto_functions);
    }

    goto_check(ns, options, goto_functions);

    // show it?
    if(cmdline.isset("show-goto-value-sets"))
    {
      value_set_analysist value_set_analysis(ns);
      value_set_analysis(goto_functions);
      std::ostringstream oss;
      show_value_sets(goto_functions, value_set_analysis, oss);
      log_result("{}", oss.str());
      return true;
    }

#if 0
    // This disabled code used to run the pointer static analysis and produce
    // pointer assertions appropriately. Disabled now that assertions are all
    // performed at symex time.
    status("Pointer Analysis");

    status("Adding Pointer Checks");

    // add pointer checks
    pointer_checks(
      goto_functions, ns, context, options, value_set_analysis);
#endif

    // remove skips
    remove_skip(goto_functions);

    // remove unreachable code
    Forall_goto_functions(f_it, goto_functions)
      remove_unreachable(f_it->second.body);

    // remove skips
    remove_skip(goto_functions);

    // recalculate numbers, etc.
    goto_functions.update();

    // add loop ids
    goto_functions.compute_loop_numbers();

    if(cmdline.isset("data-races-check"))
    {
      log_status("Adding Data Race Checks");

      value_set_analysist value_set_analysis(ns);
      value_set_analysis(goto_functions);

      add_race_assertions(value_set_analysis, context, goto_functions);

      value_set_analysis.update(goto_functions);
    }

    // show it?
    if(cmdline.isset("show-loops"))
    {
      show_loop_numbers(goto_functions);
      return true;
    }

    if(cmdline.isset("output-goto"))
    {
      log_status("Writing GOTO program to file");
      std::ofstream oss(
        cmdline.getval("output-goto"), std::ios::out | std::ios::binary);
      if(write_goto_binary(oss, context, goto_functions))
      {
        log_error("Failed to generate goto binary file"); // TODO: explain why
        abort();
      };
      return true;
    }

    // show it?
    if(
      cmdline.isset("goto-functions-too") ||
      cmdline.isset("goto-functions-only"))
    {
      std::ostringstream oss;
      goto_functions.output(ns, oss);
      log_status("{}", oss.str());
      if(cmdline.isset("goto-functions-only"))
        return true;
    }
  }

  catch(const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch(const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  catch(std::bad_alloc &)
  {
    log_error("Out of memory");
    return true;
  }

  return false;
}

int esbmc_parseoptionst::do_bmc(bmct &bmc)
{ // do actual BMC

  log_status("Starting Bounded Model Checking");

  smt_convt::resultt res = bmc.start_bmc();
  if(res == smt_convt::P_ERROR)
    abort();

#ifdef HAVE_SENDFILE_ESBMC
  if(bmc.options.get_bool_option("memstats"))
  {
    int fd = open("/proc/self/status", O_RDONLY);
    sendfile(2, fd, nullptr, 100000);
    close(fd);
  }
#endif

  return res;
}

void esbmc_parseoptionst::help()
{
  log_status("\n* * *           ESBMC {}          * * *", ESBMC_VERSION);
  std::ostringstream oss;
  oss << cmdline.cmdline_options;
  log_status("{}", oss.str());
}
