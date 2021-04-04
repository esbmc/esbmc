/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_CMDLINE_H
#define CPROVER_CMDLINE_H

#include <list>
#include <string>
#include <vector>
#include <boost/program_options.hpp>


class cmdlinet
{
public:
  void parse(int argc, const char **argv);
  const char *getval(const char *option) const;
  const std::list<std::string> &get_values(const char *option) const;

  bool isset(const char *option) const;

  void clear();

  typedef std::vector<std::string> argst;
  argst args;
   boost::program_options::variables_map vm;
   boost::program_options::options_description cmdline_options;
  cmdlinet() = default;
  ~cmdlinet();

};

#endif
