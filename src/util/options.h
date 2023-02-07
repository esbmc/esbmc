#ifndef CPROVER_OPTIONS_H
#define CPROVER_OPTIONS_H

#include <list>
#include <map>
#include <string>
#include <util/cmdline.h>

class optionst
{
public:
  typedef std::map<std::string, std::string> option_mapt;

  option_mapt option_map; // input

  const std::string get_option(const std::string &option) const;
  bool get_option(const std::string &option, std::string &value) const;
  bool get_bool_option(const std::string &option) const;
  void set_option(const std::string &option, const bool value);
  void set_option(const std::string &option, const char *value);
  void set_option(const std::string &option, const int value);
  void set_option(const std::string &option, const std::string &value);

  void cmdline(cmdlinet &cmds);

  bool is_kind() const;
};

#endif
