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

namespace po = boost::program_options;
#include <iterator>
// A helper function to print a vector.
template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
  copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
  return os;
}
enum opt_types
{
  switc,
  number,
  string
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
  void parse(int argc, const char **argv);
  const char *getval(char option) const;
  const char *getval(const char *option) const;
  const std::list<std::string> &get_values(const char *option) const;

  bool isset(const char *option) const;

  void clear();

  typedef std::vector<std::string> argst;
  argst args;
  std::string failing_option;
  po::variables_map vm;
  po::options_description cmdline_options;
  cmdlinet() = default;
  ~cmdlinet();

};

#endif
