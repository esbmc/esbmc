/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <cstdlib>

#include <util/cmdline.h>
#include <sstream>
#include <util/message/format.h>

/* Parses 's' according to a simple interpretation of shell rules, taking only
 * whitespace and the characters ', " and \ into account. */
static std::vector<std::string>
simple_shell_unescape(const char *s, const messaget &msg, const char *var)
{
  static const char WHITE[] = " \t\r\n\f\v";

  if(!s)
    return {};
  std::vector<std::string> split;
  while(*s)
  {
    /* skip white-space */
    while(*s && strchr(WHITE, *s))
      s++;
    if(!*s)
      break;
    std::string arg;
    enum : char
    {
      NONE,
      DQUOT = '"',
      SQUOT = '\'',
      ESC = '\\',
    } mode = NONE;
    while(*s)
    {
      switch(mode)
      {
      case NONE:
        /* white-space delimits strings */
        if(strchr(WHITE, *s))
          goto done;
        /* special chars in this mode */
        switch(*s)
        {
        case '\'':
          mode = SQUOT;
          s++;
          continue;
        case '"':
          mode = DQUOT;
          s++;
          continue;
        case '\\':
          /* skip first backslash */
          mode = ESC;
          s++;
          if(!*s)
            goto done;
          mode = NONE;
          break;
        }
        break;
      case SQUOT:
        /* the only special char in single-quote mode is ' */
        switch(*s)
        {
        case '\'':
          mode = NONE;
          s++;
          continue;
        }
        break;
      case DQUOT:
        /* special chars in double-quote mode */
        switch(*s)
        {
        case '"':
          mode = NONE;
          s++;
          continue;
        case '\\':
          mode = ESC;
          if(!s[1])
            goto done;
          mode = DQUOT;
          if(strchr("\\\"", s[1]))
            s++;
          break;
        }
        break;
      case ESC:
        msg.error("Arrived at an unreachable place");
        abort();
      }
      arg.push_back(*s++);
    }
  done:
    if(mode)
    {
      msg.warning(fmt::format(
        "cannot parse environment variable {}: unfinished {}, ignoring...",
        var,
        mode));
      return {};
    }
    split.emplace_back(std::move(arg));
  }
  return split;
}

std::string verification_file;

cmdlinet::~cmdlinet()
{
  clear();
}

void cmdlinet::clear()
{
  vm.clear();
  args.clear();
  options_map.clear();
}

bool cmdlinet::isset(const char *option) const
{
  return vm.count(option) > 0;
}

const std::list<std::string> &cmdlinet::get_values(const char *option) const
{
  cmdlinet::options_mapt::const_iterator value = options_map.find(option);
  assert(value != options_map.end());
  return value->second;
}

const char *cmdlinet::getval(const char *option) const
{
  cmdlinet::options_mapt::const_iterator value = options_map.find(option);
  if(value == options_map.end())
  {
    return (const char *)nullptr;
  }
  if(value->second.empty())
  {
    return (const char *)nullptr;
  }
  return value->second.front().c_str();
}

bool cmdlinet::parse(
  int argc,
  const char **argv,
  const struct group_opt_templ *opts)
{
  clear();
  unsigned int i = 0;
  for(; opts[i].groupname != "end"; i++)
  {
    boost::program_options::options_description op_desc(opts[i].groupname);
    std::vector<opt_templ> groupoptions = opts[i].options;
    for(std::vector<opt_templ>::iterator it = groupoptions.begin();
        it != groupoptions.end();
        ++it)
    {
      if(!it->type_default_value)
      {
        op_desc.add_options()(it->optstring, it->description);
      }
      else
      {
        op_desc.add_options()(
          it->optstring, it->type_default_value, it->description);
      }
    }
    cmdline_options.add(op_desc);
  }
  std::vector<opt_templ> hidden_group_options = opts[i + 1].options;
  boost::program_options::options_description hidden_cmdline_options;
  for(std::vector<opt_templ>::iterator it = hidden_group_options.begin();
      it != hidden_group_options.end() && it->optstring[0] != '\0';
      ++it)
  {
    if(!it->type_default_value)
    {
      hidden_cmdline_options.add_options()(it->optstring, "");
    }
    else
    {
      hidden_cmdline_options.add_options()(
        it->optstring, it->type_default_value, "");
    }
  }

  boost::program_options::options_description all_cmdline_options;
  all_cmdline_options.add(cmdline_options).add(hidden_cmdline_options);
  boost::program_options::positional_options_description p;
  p.add("input-file", -1);
  try
  {
    boost::program_options::store(
      boost::program_options::command_line_parser(
        simple_shell_unescape(getenv("ESBMC_OPTS"), msg, "ESBMC_OPTS"))
        .options(all_cmdline_options)
        .run(),
      vm);
    boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
        .options(all_cmdline_options)
        .positional(p)
        .run(),
      vm);
  }
  catch(std::exception &e)
  {
    msg.error(fmt::format("ESBMC error: {}", e.what()));
    return true;
  }

  if(vm.count("input-file"))
  {
    args = vm["input-file"].as<std::vector<std::string>>();
    verification_file = args.back();
  }
  for(auto &it : vm)
  {
    std::list<std::string> res;
    std::string option_name = it.first;
    const boost::any &value = vm[option_name].value();
    if(const int *v = boost::any_cast<int>(&value))
    {
      res.emplace_front(std::to_string(*v));
    }
    else if(const std::string *v = boost::any_cast<std::string>(&value))
    {
      res.emplace_front(*v);
    }
    else if(
      const std::vector<int> *v = boost::any_cast<std::vector<int>>(&value))
    {
      for(auto iter = v->begin(); iter != v->end(); ++iter)
      {
        res.emplace_front(std::to_string(*iter));
      }
    }
    else
    {
      std::vector<std::string> src =
        vm[option_name].as<std::vector<std::string>>();
      res.assign(src.begin(), src.end());
    }
    std::pair<options_mapt::iterator, bool> result =
      options_map.insert(options_mapt::value_type(option_name, res));
    if(!result.second)
      result.first->second = res;
  }
  for(std::vector<opt_templ>::iterator it = hidden_group_options.begin();
      it != hidden_group_options.end() && it->optstring[0] != '\0';
      ++it)
  {
    if(it->description[0] != '\0' && vm.count(it->description))
    {
      std::list<std::string> value = get_values(it->description);
      std::pair<options_mapt::iterator, bool> result =
        options_map.insert(options_mapt::value_type(it->optstring, value));
      if(!result.second)
        result.first->second = value;
    }
  }
  return false;
}
