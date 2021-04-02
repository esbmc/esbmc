/*******************************************************************\

Module: Options

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cstdlib>
#include <util/i2string.h>
#include <util/options.h>
// #include <iostream>
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
  // std::cout<<"Starting conversion of options\n";
  // Pump command line options into options list
  for(const auto &it : cmds.vm)
  {
    auto option_name = it.first;
// std::cout<<"Option Name before is set:"<<option_name<<"\n";
    if(cmds.isset(option_name.c_str()))
    {
    //  std::cout<<"Option Name after is set:"<<option_name<<"\n";
      if(!it.second.empty())
      {
        // std::cout<<"Second is not empty\n";
        //      std::cout<<"Option Value if not empty after is set:"<<cmds.getval(option_name.c_str())<<"\n";

        set_option(option_name, cmds.getval(option_name.c_str()));
      }
      else
      {
        set_option(option_name, true);
      }
    }
  }
}

