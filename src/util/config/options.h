#ifndef CPROVER_OPTIONS_H
#define CPROVER_OPTIONS_H

#include <list>
#include <vector>
#include <map>
#include <string>
#include <util/config/cmdline.h>

class optionst
{
public:
  typedef std::map<std::string, std::string> option_mapt;

  option_mapt option_map; // input

  /// Every value a repeatable option was given on the command line, in the
  /// order given. #option_map keeps only the last, which is what its consumers
  /// want and what a context key must not settle for: `-DA -DB` and `-DB` are
  /// different verifications (esbmc/esbmc#7143).
  std::map<std::string, std::vector<std::string>> option_values;

  const std::string get_option(const std::string &option) const;
  bool get_option(const std::string &option, std::string &value) const;
  bool get_bool_option(const std::string &option) const;
  void set_option(const std::string &option, const bool value);
  void set_option(const std::string &option, const char *value);
  void set_option(const std::string &option, const int value);
  void set_option(const std::string &option, const std::string &value);

  void cmdline(cmdlinet &cmds);

  bool is_kind() const;

  /// \brief Whether any function-contract mode is active. Contract clauses
  ///   state nothing outside these modes and are dropped, which two separate
  ///   lowering paths need to agree on.
  bool contracts_enabled() const;
};

#endif
