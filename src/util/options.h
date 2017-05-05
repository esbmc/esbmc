/*******************************************************************\

Module: Options

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

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
  
  virtual const std::string get_option(const std::string &option) const;
  virtual bool get_bool_option(const std::string &option) const;
  virtual void set_option(const std::string &option, const bool value);
  virtual void set_option(const std::string &option, const char *value);
  virtual void set_option(const std::string &option, const int value);
  virtual void set_option(const std::string &option, const std::string &value);

  virtual void cmdline(cmdlinet &cmds);
  
  optionst() { }
  virtual ~optionst() { }
};

#endif
