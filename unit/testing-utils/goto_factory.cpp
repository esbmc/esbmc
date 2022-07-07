#include "goto_factory.h"

#include <cstdio>
#include <fstream>
#include <langapi/mode.h>
#include <langapi/language_ui.h>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_check.h>
#include <goto-programs/remove_unreachable.h>
#include <goto-programs/remove_skip.h>
#include <util/cmdline.h>
#include <util/message.h>

const mode_table_et mode_table[] = {
  LANGAPI_MODE_CLANG_C,
  LANGAPI_MODE_CLANG_CPP,
  LANGAPI_MODE_END}; // This is the simplest way to have this

void goto_factory::create_file_from_istream(
  std::istream &c_inputstream,
  std::string filename)
{
  std::ofstream output(filename); // Change this for C++
  if(!output.good())
  {
    perror("Could not create output C file\n");
    exit(1);
  }

  while(c_inputstream.good())
  {
    std::string line;
    c_inputstream >> line;
    output << line;
    output << " ";
  }

  output.close();
}

void goto_factory::config_environment(
  goto_factory::Architecture arch,
  cmdlinet c,
  optionst o)
{
  config.set(c);

  switch(arch)
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

goto_functionst goto_factory::get_goto_functions(
  std::istream &c_file,
  goto_factory::Architecture arch)
{
  /*
     * 1. Create an tmp file from istream
     * 2. Parse the file using clang-frontend
     * 3. Return the result
     */

  // Create tmp file
  std::string filename(
    "tmp.c"); // TODO: Make this unique and add support for CPP
  goto_factory::create_file_from_istream(c_file, filename);

  cmdlinet cmd = goto_factory::get_default_cmdline(filename);
  optionst opts = goto_factory::get_default_options(cmd);

  goto_factory::config_environment(arch, cmd, opts);
  return goto_factory::get_goto_functions(cmd, opts);
}

goto_functionst goto_factory::get_goto_functions(
  std::istream &c_file,
  cmdlinet &cmd,
  optionst &opts,
  goto_factory::Architecture arch)
{
  /*
     * 1. Create an tmp file from istream
     * 2. Parse the file using clang-frontend
     * 3. Return the result
     */

  // Create tmp file
  std::string filename(
    "tmp.c"); // TODO: Make this unique and add support for CPP
  goto_factory::create_file_from_istream(c_file, filename);
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

bool goto_factory::parse(language_uit &l)
{
  l.context.clear();
  if(l.parse())
    return false; // TODO: This can be used to add testcases for frontend
  if(l.typecheck())
    return false;
  if(l.final())
    return false;
  l.clear_parse();
  return true;
}

goto_functionst goto_factory::get_goto_functions(cmdlinet &cmd, optionst &opts)
{
  goto_functionst goto_functions;
  language_uit lui(cmd);
  if(!goto_factory::parse(lui))
    return goto_functions;

  migrate_namespace_lookup = new namespacet(lui.context);

  goto_convert(lui.context, opts, goto_functions);

  namespacet ns(lui.context);
  goto_check(ns, opts, goto_functions);
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
  return goto_functions;
}
