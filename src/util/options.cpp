/*******************************************************************\

Module: Options

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

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

  for(std::vector<cmdlinet::optiont>::const_iterator it = cmds.options.begin();
      it != cmds.options.end();
      it++)
  {
    if(it->isset)
    {
      if(it->hasval)
      {
        if(it->islong)
        {
          set_option(it->optstring, cmds.getval(it->optstring.c_str()));
        }
        else
        {
          std::string str(&it->optchar, 1);
          set_option(str, cmds.getval(it->optchar));
        }
      }
      else
      {
        if(it->islong)
        {
          set_option(it->optstring, true);
        }
        else
        {
          std::string str(&it->optchar, 1);
          set_option(str, true);
        }
      }
    }
  }
}
