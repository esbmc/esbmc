#include <solvers/bitwuzllob/bitwuzllob_conv.h>
#include <util/filesystem.h>
#include <util/message.h>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <list>
#include <optional>

#ifndef _WIN32
#  include <sys/wait.h>
#  include <unistd.h>
#endif

static std::string prog_command(const optionst &options)
{
  std::string cmd = options.get_option("bitwuzllob-prog");
  return cmd.empty() ? "mallob -mono=%f -mono-app=SMT" : cmd;
}

smt_solver_baset *create_new_bitwuzllob_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api [[maybe_unused]],
  array_iface **array_api,
  fp_convt **fp_api)
{
  /* Mallob's mono mode processes a single task and terminates; strategies
   * that solve repeatedly or incrementally cannot be served by it. */
  static const char *incompatible[] = {
    "incremental-bmc",
    "falsification",
    "k-induction",
    "k-induction-parallel",
    "termination",
    "smt-during-symex",
    "multi-property",
    "parallel-solving"};
  for (const char *opt : incompatible)
    if (options.get_bool_option(opt))
    {
      log_error(
        "the bitwuzllob backend runs Mallob in one-shot mono mode and does "
        "not support --{}; use a linked solver (e.g. --bitwuzla) for "
        "incremental strategies",
        opt);
      abort();
    }

  bitwuzllob_convt *conv = new bitwuzllob_convt(ns, options);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

/* Whether the formula file is a temporary this backend creates, as opposed
 * to a user-supplied --output path (or stdout under --smt-formula-only). */
static bool uses_temp_formula(const optionst &options)
{
  std::string output = options.get_option("output");
  return !options.get_bool_option("smt-formula-only") &&
         (output.empty() || output == "-");
}

/* The formula file Mallob is pointed at. Honour --output so the user can keep
 * the formula; otherwise use a self-cleaning temporary. With
 * --smt-formula-only no solver runs, so honour --output including stdout via
 * "-" (the default when no file is given). */
static std::string choose_formula_path(const optionst &options)
{
  std::string output = options.get_option("output");
  if (options.get_bool_option("smt-formula-only"))
    return output.empty() ? "-" : output;
  if (!uses_temp_formula(options))
    return output;

  if (output == "-")
    log_warning(
      "bitwuzllob: ignoring --output -: Mallob reads the formula from a "
      "file, not stdout; use --output <filename> to keep the formula");

  file_operations::tmp_file tmp =
    file_operations::create_tmp_file("esbmc-bitwuzllob-%%%%-%%%%.smt2");
  std::string path = tmp.path();
  /* Keep the file: the smtlib file_emitter reopens it for writing; it is
   * removed in our destructor, or by cleanup_registered_tmps() on the
   * signal/timeout exit paths that skip destructors. */
  fclose(tmp.file());
  tmp.keep(true);
  file_operations::register_tmp_for_cleanup(path);
  return path;
}

bitwuzllob_convt::bitwuzllob_convt(
  const namespacet &ns,
  const optionst &options)
  : bitwuzllob_convt(ns, options, choose_formula_path(options))
{
}

/* With --result-only no counterexample is ever built (bmc.cpp skips trace
 * construction), so feeding the formula to a local model solver would only
 * start a full parallel solve whose answer is never read. */
static std::string model_prog(const optionst &options)
{
  std::string prog = options.get_option("bitwuzllob-model-prog");
  if (!prog.empty() && options.get_bool_option("result-only"))
  {
    log_warning(
      "bitwuzllob: ignoring --bitwuzllob-model-prog: --result-only never "
      "builds a counterexample, so no model solver is needed");
    return "";
  }
  return prog;
}

bitwuzllob_convt::bitwuzllob_convt(
  const namespacet &ns,
  const optionst &options,
  const std::string &_formula_path)
  : smtlib_convt(ns, options, model_prog(options), _formula_path),
    formula_path(_formula_path)
{
}

bitwuzllob_convt::~bitwuzllob_convt()
{
  if (uses_temp_formula(options))
    remove(formula_path.c_str());
}

std::string bitwuzllob_convt::dump_smt()
{
  /* Under --smt-formula-only no solve follows; complete the dump with the
   * (check-sat) like the base class. Under --smt-formula-too our dec_solve()
   * emits the (check-sat) itself: appending one here as well would hand
   * Mallob a formula containing two. The base implementation also reports
   * the destination from the --output option, which this backend redirects
   * to the formula file. */
  if (options.get_bool_option("smt-formula-only"))
    return smtlib_convt::dump_smt();
  log_status("SMT formula written to {}", formula_path);
  return "SMT formula dumped successfully";
}

smt_resultt bitwuzllob_convt::dec_solve()
{
  if (solved)
  {
    log_error(
      "the bitwuzllob backend supports a single (check-sat) query per run; "
      "incremental strategies are not supported");
    abort();
  }
  solved = true;

  pre_solve();

  /* The (check-sat) goes to both sinks: the formula file for Mallob, and the
   * local model solver's pipe (if configured), which starts solving in
   * parallel and only gets waited for when a model is actually needed. */
  emit_check_sat();

  smt_resultt res = run_bitwuzllob();
  if (res != P_SATISFIABLE)
  {
    /* No model will be read; stop the local solver we fed in parallel rather
     * than let it keep solving until this object is destroyed. */
    emit_proc.terminate();
    return res;
  }

  if (!emit_proc)
  {
    if (options.get_bool_option("result-only"))
      return P_SATISFIABLE;
    log_error(
      "bitwuzllob: formula is satisfiable, but building the counterexample "
      "requires a local interactive SMT-LIB2 solver; re-run with "
      "--bitwuzllob-model-prog <cmd> (e.g. \"z3 -in\") or with --result-only");
    abort();
  }

  smt_resultt model_res = read_check_sat_response();
  if (model_res != P_SATISFIABLE)
  {
    log_error(
      "bitwuzllob: Bitwuzllob reported sat but the local model solver did "
      "not; refusing to build a counterexample from a diverging model");
    abort();
  }
  return P_SATISFIABLE;
}

/* Recognize a verdict on one (whitespace-trimmed) output line. Accepts the
 * bare SMT-LIB verdicts as well as the SAT-competition-style "s ..." lines
 * Mallob emits, amid arbitrary log noise. */
static std::optional<smt_resultt> parse_verdict_line(const std::string &line)
{
  if (line == "sat" || line == "s SATISFIABLE")
    return P_SATISFIABLE;
  if (line == "unsat" || line == "s UNSATISFIABLE")
    return P_UNSATISFIABLE;
  if (line == "unknown" || line == "s UNKNOWN")
    return P_ERROR;
  return {};
}

/* pclose() returns a waitpid()-style status word, not the process exit code;
 * decode it so logs show "exit code 10" rather than 2560. */
static std::string describe_exit_status(int status)
{
#ifndef _WIN32
  if (WIFEXITED(status))
    return "exit code " + std::to_string(WEXITSTATUS(status));
  if (WIFSIGNALED(status))
    return "signal " + std::to_string(WTERMSIG(status));
#endif
  return "status " + std::to_string(status);
}

smt_resultt bitwuzllob_convt::run_bitwuzllob()
{
  std::string cmd = prog_command(options);

  /* Substitute the formula file for every %f, or append it. The path is
   * single-quoted for the shell; embedded single quotes (possible via
   * --output) are escaped as '\''. */
  std::string escaped_path = formula_path;
  for (size_t q = escaped_path.find('\''); q != std::string::npos;
       q = escaped_path.find('\'', q + 4))
    escaped_path.replace(q, 1, "'\\''");
  std::string quoted_path = "'" + escaped_path + "'";
  size_t pos = cmd.find("%f");
  if (pos == std::string::npos)
    cmd += " " + quoted_path;
  else
    for (; pos != std::string::npos; pos = cmd.find("%f", pos))
    {
      cmd.replace(pos, 2, quoted_path);
      pos += quoted_path.size();
    }

  log_status("Running Bitwuzllob: {}", cmd);

#ifdef _WIN32
  FILE *out = popen(cmd.c_str(), "r");
  if (!out)
  {
    log_error("bitwuzllob: failed to run \"{}\": {}", cmd, strerror(errno));
    abort();
  }
#else
  /* Spawn the solver in its own process group so that, on an ESBMC timeout
   * or fatal signal, the exit handlers can kill the whole subtree (a wrapper
   * shell and any mpirun ranks) in one killpg — plain popen() leaves the
   * child in ESBMC's own group, which the timeout handler cannot target
   * without killing itself. */
  int fds[2];
  if (pipe(fds) != 0)
  {
    log_error("bitwuzllob: pipe() failed: {}", strerror(errno));
    abort();
  }
  pid_t pid = fork();
  if (pid < 0)
  {
    log_error("bitwuzllob: fork() failed: {}", strerror(errno));
    abort();
  }
  if (pid == 0)
  {
    setpgid(0, 0); // become leader of a new group; pgid == pid
    close(fds[0]);
    if (fds[1] != STDOUT_FILENO)
    {
      dup2(fds[1], STDOUT_FILENO);
      close(fds[1]);
    }
    const char *shell = getenv("SHELL");
    if (!shell || !*shell)
      shell = "sh";
    execlp(shell, shell, "-c", cmd.c_str(), static_cast<char *>(nullptr));
    _exit(127);
  }
  setpgid(pid, pid); // also set from the parent to close the fork/exec race
  close(fds[1]);
  const long child_pgid = pid;
  file_operations::register_pgroup_for_cleanup(child_pgid);
  FILE *out = fdopen(fds[0], "r");
  if (!out)
  {
    log_error("bitwuzllob: fdopen() failed: {}", strerror(errno));
    abort();
  }
#endif

  /* Scan the output for verdict lines; the last one wins. Keep a small tail
   * of the output for diagnostics. */
  std::optional<smt_resultt> verdict;
  std::list<std::string> tail;
  char buf[4096];
  while (fgets(buf, sizeof(buf), out))
  {
    std::string line(buf);
    while (!line.empty() && isspace((unsigned char)line.back()))
      line.pop_back();
    size_t start = line.find_first_not_of(" \t");
    if (start != std::string::npos)
      line.erase(0, start);

    if (std::optional<smt_resultt> v = parse_verdict_line(line))
      verdict = v;

    tail.push_back(std::move(line));
    if (tail.size() > 20)
      tail.pop_front();
  }

#ifdef _WIN32
  int status = pclose(out);
#else
  fclose(out);
  int status = 0;
  waitpid(pid, &status, 0);
  file_operations::unregister_pgroup(child_pgid);
#endif

#ifndef _WIN32
  /* A verdict from a solver that died on a signal (crash, OOM kill, an
   * mpirun rank failure tearing the job down) cannot be trusted: discard it
   * rather than turn a truncated run into a verification verdict. Non-zero
   * *exit codes* stay accepted — SAT-competition style solvers exit 10/20. */
  if (verdict && WIFSIGNALED(status))
  {
    log_error(
      "bitwuzllob: solver command \"{}\" died with {}; discarding its "
      "verdict",
      cmd,
      describe_exit_status(status));
    return P_ERROR;
  }
#endif

  if (!verdict)
  {
    std::string tail_str;
    for (const std::string &line : tail)
      tail_str += "\n  " + line;
    log_error(
      "bitwuzllob: no sat/unsat verdict in the output of \"{}\" ({}); last "
      "output lines:{}",
      cmd,
      describe_exit_status(status),
      tail_str);
    return P_ERROR;
  }

  /* Non-zero exit is expected with SAT-competition-style codes (10/20), so
   * only surface it at debug level alongside the parsed verdict. */
  if (status != 0)
    log_debug(
      "bitwuzllob",
      "solver command exited with {} after verdict",
      describe_exit_status(status));

  if (*verdict == P_ERROR)
    log_error("bitwuzllob: solver returned unknown");
  return *verdict;
}

const std::string bitwuzllob_convt::solver_text()
{
  return "Bitwuzllob '" + prog_command(options) + "'";
}
