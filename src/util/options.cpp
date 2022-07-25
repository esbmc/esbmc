#include <cstdlib>
#include <util/i2string.h>
#include <util/options.h>

void optionst::set_option(const std::string &option, const std::string &value)
{
  std::pair<option_mapt::iterator, bool> result =
    option_map.insert(option_mapt::value_type(option, value));

  if(!result.second)
    result.first->second = value;
}

void optionst::set_option(const std::string &option, const char *value)
{
  set_option(option, std::string(value));
}

void optionst::set_option(const std::string &option, const bool value)
{
  set_option(option, std::string(value ? "1" : "0"));
}

void optionst::set_option(const std::string &option, const int value)
{
  set_option(option, i2string(value));
}

bool optionst::get_bool_option(const std::string &option) const
{
  return atoi(get_option(option).c_str());
}

const std::string optionst::get_option(const std::string &option) const
{
  std::map<std::string, std::string>::const_iterator it =
    option_map.find(option);

  if(it != option_map.end())
    return it->second;

  return "";
}

void optionst::cmdline(cmdlinet &cmds)
{
  // Pump command line options into options list
  for(auto &it : cmds.vm)
  {
    std::string option_name = it.first;
    if(cmds.isset(option_name.c_str()) && !it.second.defaulted())
    {
      const char *value = cmds.getval(option_name.c_str());
      bool hasArgument = *value != 0;
      if(hasArgument)
        set_option(option_name, value);
      else
        set_option(option_name, true);
    }
  }
}

bool optionst::is_kind() const
{
  return get_bool_option("k-induction") ||
         get_bool_option("k-induction-parallel");
}
