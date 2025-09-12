#include <symbol_id.h>
#include <regex>
#include <sstream>

symbol_id::symbol_id(
  const std::string &file,
  const std::string &clazz,
  const std::string &function)
  : filename_(file), classname_(clazz), function_name_(function)
{
}

symbol_id symbol_id::from_string(const std::string &str)
{
  symbol_id id;

  const std::string prefix = "py:";
  // Check if the string starts with the expected prefix
  if (str.rfind(prefix, 0) != 0)
    return id; // Invalid prefix, return empty symbol_id

  id.set_prefix(prefix);

  // Remove the prefix "py:" from the string
  std::string s = str.substr(prefix.length());

  // The filename comes before the first '@'
  size_t at_pos = s.find('@');
  if (at_pos == std::string::npos)
  {
    id.set_filename(s); // Only filename provided
    return id;
  }

  // Set the filename
  id.set_filename(s.substr(0, at_pos));

  // Extract the remainder (e.g., @C@ClassName@F@FunctionName)
  s = s.substr(at_pos);

  // Regex to extract class name: @C@ClassName
  std::regex class_regex(R"(@C@([^@]+))");
  std::smatch match;
  if (std::regex_search(s, match, class_regex))
  {
    id.set_class(match[1].str());
  }

  // Regex to extract function name: @F@FunctionName
  std::regex func_regex(R"(@F@([^@]+))");
  if (std::regex_search(s, match, func_regex))
  {
    id.set_function(match[1].str());
  }

  return id;
}

std::string symbol_id::to_string() const
{
  std::stringstream ss;
  ss << prefix_ << filename_;

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

std::string symbol_id::global_to_string() const
{
  std::stringstream ss;
  ss << prefix_ << filename_;

  if (!object_.empty())
    ss << "@" << object_;

  return ss.str();
}

void symbol_id::clear()
{
  filename_.clear();
  classname_.clear();
  function_name_.clear();
  object_.clear();
  attribute_.clear();
}
