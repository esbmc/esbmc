#include <cstdlib>
#include <util/i2string.h>
#include <util/options.h>

void optionst::set_option(const std::string &option, const std::string &value)
{
  std::pair<option_mapt::iterator, bool> result =
    option_map.insert(option_mapt::value_type(option, value));

  if (!result.second)
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

  if (it != option_map.end())
    return it->second;

  return "";
}

bool optionst::get_option(const std::string &option, std::string &value) const
{
  auto it = option_map.find(option);
  if (it == option_map.end())
    return false;
  value = it->second;
  return true;
}

void optionst::cmdline(cmdlinet &cmds)
{
  // Pump command line options into options list
  for (const auto &it : cmds.vm)
  {
    const auto option_name = it.first;
    if (cmds.isset(option_name.c_str()) && !it.second.defaulted())
    {
      std::string value_str;
      for (const auto &value : cmds.get_values(option_name.c_str()))
        if (!value.empty())
          value_str.append(value).append(" ");

      if (value_str.empty())
        set_option(option_name, true);
      else
        set_option(option_name, value_str);
    }
  }
}

bool optionst::is_kind() const
{
  return get_bool_option("k-induction") ||
         get_bool_option("k-induction-parallel");
}
