// Helper for tests that uses goto
#pragma once
#include <istream>
#include <goto-programs/goto_functions.h>
#include <langapi/language_ui.h>

class program : public language_uit
{
public:
  goto_functionst functions;

  program() = default;
  program(program &&) = default;
  program &operator=(program &&) = default;
};

/**
 * @brief This class parses C inputs
 * and generates goto-functions
 * 
 */
class goto_factory
{
public:
  enum class Architecture
  {
    BIT_16,
    BIT_32,
    BIT_64
  };

  /**
   * @brief Get the goto functions object
   *
   * @param c_inputstream input stream containing the C program
   * @return goto_functionst of the parsed object
   */
  static program get_goto_functions(
    std::istream &c_inputstream,
    goto_factory::Architecture arch = goto_factory::Architecture::BIT_16);

  static program get_goto_functions(
    std::string &str,
    goto_factory::Architecture arch = goto_factory::Architecture::BIT_16);

  static program get_goto_functions(
    std::istream &c_inputstream,
    cmdlinet &cmd,
    optionst &opts,
    goto_factory::Architecture arch = goto_factory::Architecture::BIT_16);

  static cmdlinet get_default_cmdline(const std::string filename);
  static optionst get_default_options(cmdlinet cmd);

private:
  static bool parse(const cmdlinet &cmdline, language_uit &l);
  static void
  create_file_from_istream(std::istream &c_inputstream, std::string filename);
  static void create_file_from_string(std::string &str, std::string filename);

  static void
  config_environment(goto_factory::Architecture arch, cmdlinet c, optionst o);

  static program get_goto_functions(cmdlinet &cmd, optionst &opts);
};
