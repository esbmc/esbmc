#include <solvers/oneshot_options.h>
#include <util/base/filesystem.h>
#include <util/config/options.h>
#include <util/message/message.h>

#include <cstdio>

namespace oneshot_options
{
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
  /* Keep the file: camada's file emitter reopens it for writing; it is
   * removed in the backend's destructor, or by cleanup_registered_tmps() on
   * the signal/timeout exit paths that skip destructors. */
  fclose(tmp.file());
  tmp.keep(true);
  file_operations::register_tmp_for_cleanup(path);
  return path;
}

std::string model_prog(const optionst &options, const char *name)
{
  std::string prog = options.get_option("smtlib-oneshot-model-prog");
  if (!prog.empty() && options.get_bool_option("result-only"))
  {
    log_warning(
      "{}: ignoring --smtlib-oneshot-model-prog: --result-only never builds a "
      "counterexample, so no model solver is needed",
      name);
    return "";
  }
  return prog;
}
} // namespace oneshot_options
