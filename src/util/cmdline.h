#ifndef CPROVER_CMDLINE_H
#define CPROVER_CMDLINE_H

#include <list>
#include <string>
#include <vector>
#include <map>
#include <boost/program_options.hpp>

/* WORKAROUND: On *BSD macOS the include of some system headers
 * makes the definition of isset as a macro, which conflicts with
 * the member */
#ifdef isset
#undef isset
#endif
struct opt_templ
{
  const char *optstring;
  const boost::program_options::value_semantic *type_default_value;
  const char *description;
};

struct group_opt_templ
{
  std::string groupname;
  std::vector<opt_templ> options;
};

class cmdlinet
{
public:
  bool parse(int argc, const char **argv, const struct group_opt_templ *opts);
  const char *getval(const char *option) const;
  const std::list<std::string> &get_values(const char *option) const;
  bool isset(const char *option) const;
  void clear();
  typedef std::vector<std::string> argst;
  argst args;
  boost::program_options::variables_map vm;
  boost::program_options::options_description cmdline_options;
  cmdlinet()
  {
  }
  ~cmdlinet();
  typedef std::map<std::string, std::list<std::string>> options_mapt;
  options_mapt options_map;
};

#endif
