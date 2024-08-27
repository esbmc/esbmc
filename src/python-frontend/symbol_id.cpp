#include <symbol_id.h>
#include <sstream>

symbol_id::symbol_id(
  const std::string &file,
  const std::string &clazz,
  const std::string &function)
  : filename_(file), classname_(clazz), function_name_(function)
{
}

std::string symbol_id::to_string() const
{
  std::stringstream ss;
  ss << "py:" << filename_;

  if (!classname_.empty())
    ss << "@C@" << classname_;

  if (!function_name_.empty())
    ss << "@F@" << function_name_;

  if (!object_.empty())
    ss << "@" << object_;

  if (!attribute_.empty())
    ss << "@" << attribute_;

  return ss.str();
}
