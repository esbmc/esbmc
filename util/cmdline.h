/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_CMDLINE_H
#define CPROVER_CMDLINE_H

#include <vector>
#include <list>
#include <string>

enum opt_types {
  switc, number, string
};

struct opt_templ
{
  char optchar;
  std::string optstring;
  opt_types type;
  std::string init;
};

class cmdlinet
{
public:
  bool parse(int argc, const char **argv, const struct opt_templ *opts);
  const char *getval(char option) const;
  const char *getval(const char *option) const;
  const std::list<std::string> &get_values(const char *option) const;
  const std::list<std::string> &get_values(char option) const;
  bool isset(char option) const;
  bool isset(const char *option) const;

  void clear();

  typedef std::vector<std::string> argst;
  argst args;
  std::string failing_option;
  
  cmdlinet();
  ~cmdlinet();
  
protected:
  struct optiont
  {
    bool isset, hasval, islong;
    char optchar;
    std::string optstring;
    std::list<std::string> values;
  };
   
  std::vector<optiont> options;

  int getoptnr(char option) const;
  int getoptnr(const char *option) const;

  friend class optionst;
};

#endif
