#include "goto_factory.h"

#include <cstdio>
#include <fstream>
#include <langapi/mode.h>
#include <langapi/language_ui.h>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_check.h>
#include <goto-programs/remove_unreachable.h>
#include <goto-programs/remove_no_op.h>
#include <util/cmdline.h>
#include <util/message.h>
#include <util/filesystem.h>

const mode_table_et mode_table[] = {
  LANGAPI_MODE_CLANG_C,
  LANGAPI_MODE_CLANG_CPP,
  LANGAPI_MODE_END}; // This is the simplest way to have this

void goto_factory::create_file_from(
  std::istream &c_inputstream,
  std::string filename)
{
  std::ofstream output(filename); // Change this for C++
  if (!output.good())
  {
    perror("Could not create output C file\n");
    exit(1);
  }

  while (c_inputstream.good())
  {
    std::string line;
    c_inputstream >> line;
    output << line;
    output << " ";
  }

  output.close();
}

void goto_factory::create_file_from(std::string &str, std::string filename)
{
  std::ofstream output(filename); // Change this for C++
  if (!output.good())
  {
    perror("Could not create output C file\n");
    exit(1);
  }

  output << str;

  output.close();
}

void goto_factory::config_environment(
  goto_factory::Architecture arch,
  cmdlinet c,
  optionst o)
{
  config.set(c);

  switch (arch)
  {
  case goto_factory::Architecture::BIT_16:
    config.ansi_c.set_data_model(configt::LP32);
    break;

  case goto_factory::Architecture::BIT_32:
    config.ansi_c.set_data_model(configt::ILP32);
    break;

  case goto_factory::Architecture::BIT_64:
    config.ansi_c.set_data_model(configt::LP64);
    break;
  default:
    break;
  }

  config.options = o;
}

program goto_factory::get_goto_functions(
  std::istream &c_file,
  goto_factory::Architecture arch,
  const std::string &test_name)
{
  std::string filename(
    file_operations::get_unique_tmp_path("esbmc-test-%%%%%%"));
  filename += "/" + test_name;
  log_status("Creating {}", filename);
  goto_factory::create_file_from(c_file, filename);

  return goto_factory::get_goto_functions_internal(filename, arch);
}

program goto_factory::get_goto_functions(
  std::string &str,
  goto_factory::Architecture arch,
  const std::string &test_name)
{
  std::string filename(
    file_operations::get_unique_tmp_path("esbmc-test-%%%%%%"));
  filename += "/" + test_name;
  goto_factory::create_file_from(str, filename);

  return goto_factory::get_goto_functions_internal(filename, arch);
}

program goto_factory::get_goto_functions_internal(
  const std::string &filename,
  goto_factory::Architecture arch)
{
  cmdlinet cmd = goto_factory::get_default_cmdline(filename);
  optionst opts = goto_factory::get_default_options(cmd);

  goto_factory::config_environment(arch, cmd, opts);
  return goto_factory::get_goto_functions(cmd, opts);
}

cmdlinet goto_factory::get_default_cmdline(const std::string filename)
{
  cmdlinet cmdline;
  cmdline.args.push_back(filename);
  return cmdline;
}

optionst goto_factory::get_default_options(cmdlinet cmd)
{
  optionst options;
  options.cmdline(cmd);
  options.set_option("floatbv", true);
  options.set_option("context-bound", -1);
  options.set_option("deadlock-check", false);
  return options;
}

bool goto_factory::parse(const cmdlinet &cmdline, language_uit &l)
{
  l.context.clear();
  if (l.parse(cmdline))
    return false; // TODO: This can be used to add testcases for frontend
  if (l.typecheck())
    return false;
  if (l.final())
    return false;
  l.clear_parse();
  return true;
}

program goto_factory::get_goto_functions(cmdlinet &cmd, optionst &opts)
{
  program P;

  if (goto_factory::parse(cmd, P))
  {
    goto_functionst &goto_functions = P.functions;
    goto_convert(P.context, opts, goto_functions);

    namespacet &ns = P.ns;
    goto_check(ns, opts, goto_functions);
    // remove no-op's
    remove_no_op(goto_functions);

    // Remove unreachable code
    remove_unreachable(goto_functions);

    // recalculate numbers, etc.
    goto_functions.update();

    // add loop ids
    goto_functions.compute_loop_numbers();
  }

  return P;
}
