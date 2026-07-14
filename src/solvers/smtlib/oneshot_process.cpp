#include <solvers/smtlib/oneshot_process.h>
#include <util/filesystem.h>
#include <util/message.h>
#include <util/options.h>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <list>

#ifndef _WIN32
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace oneshot_process
{
std::optional<smt_resultt> parse_verdict_line(const std::string &raw)
{
  size_t start = raw.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return {};
  size_t end = raw.find_last_not_of(" \t\r\n");
  std::string line = raw.substr(start, end - start + 1);

  if (line == "sat" || line == "s SATISFIABLE")
    return P_SATISFIABLE;
  if (line == "unsat" || line == "s UNSATISFIABLE")
    return P_UNSATISFIABLE;
  if (line == "unknown" || line == "s UNKNOWN")
    return P_ERROR;
  return {};
}

bool uses_temp_formula(const optionst &options)
{
  std::string output = options.get_option("output");
  return !options.get_bool_option("smt-formula-only") &&
         (output.empty() || output == "-");
}

std::string choose_formula_path(const optionst &options, const char *name)
{
  std::string output = options.get_option("output");
  if (options.get_bool_option("smt-formula-only"))
    return output.empty() ? "-" : output;
  if (!uses_temp_formula(options))
    return output;

  if (output == "-")
    log_warning(
      "{}: ignoring --output -: the solver program reads the formula from a "
      "file, not stdout; use --output <filename> to keep the formula",
      name);

  file_operations::tmp_file tmp = file_operations::create_tmp_file(
    std::string("esbmc-") + name + "-%%%%-%%%%.smt2");
  std::string path = tmp.path();
  /* Keep the file: the smtlib file_emitter reopens it for writing; it is
   * removed in the backend's destructor, or by cleanup_registered_tmps() on
   * the signal/timeout exit paths that skip destructors. */
  fclose(tmp.file());
  tmp.keep(true);
  file_operations::register_tmp_for_cleanup(path);
  return path;
}

std::string model_prog(const optionst &options, const char *name)
{
  std::string prog = options.get_option(std::string(name) + "-model-prog");
  if (!prog.empty() && options.get_bool_option("result-only"))
  {
    log_warning(
      "{}: ignoring --{}-model-prog: --result-only never builds a "
      "counterexample, so no model solver is needed",
      name,
      name);
    return "";
  }
  return prog;
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

smt_resultt run_solver(
  const std::string &cmd_template,
  const std::string &formula_path,
  const char *name)
{
  std::string cmd = cmd_template;

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

  log_status("Running {}: {}", name, cmd);

#ifdef _WIN32
  FILE *out = popen(cmd.c_str(), "r");
  if (!out)
  {
    log_error("{}: failed to run \"{}\": {}", name, cmd, strerror(errno));
    abort();
  }
#else
  /* Spawn the solver in its own process group so that, on an ESBMC timeout
   * or fatal signal, the exit handlers can kill the whole subtree (a wrapper
   * shell and any further children) in one killpg — plain popen() leaves the
   * child in ESBMC's own group, which the timeout handler cannot target
   * without killing itself. */
  int fds[2];
  if (pipe(fds) != 0)
  {
    log_error("{}: pipe() failed: {}", name, strerror(errno));
    abort();
  }
  pid_t pid = fork();
  if (pid < 0)
  {
    log_error("{}: fork() failed: {}", name, strerror(errno));
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
    log_error("{}: fdopen() failed: {}", name, strerror(errno));
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
  /* A verdict from a solver that died on a signal (crash, OOM kill, a
   * wrapper tearing the job down) cannot be trusted: discard it rather than
   * turn a truncated run into a verification verdict. Non-zero *exit codes*
   * stay accepted — SAT-competition style solvers exit 10/20. */
  if (verdict && WIFSIGNALED(status))
  {
    log_error(
      "{}: solver command \"{}\" died with {}; discarding its verdict",
      name,
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
      "{}: no sat/unsat verdict in the output of \"{}\" ({}); last "
      "output lines:{}",
      name,
      cmd,
      describe_exit_status(status),
      tail_str);
    return P_ERROR;
  }

  /* Non-zero exit is expected with SAT-competition-style codes (10/20), so
   * only surface it at debug level alongside the parsed verdict. */
  if (status != 0)
    log_debug(
      name,
      "solver command exited with {} after verdict",
      describe_exit_status(status));

  if (*verdict == P_ERROR)
    log_error("{}: solver returned unknown", name);
  return *verdict;
}
} // namespace oneshot_process
