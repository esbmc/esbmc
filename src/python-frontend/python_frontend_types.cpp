#include <python_frontend_types.h>
#include <map>
#include <string>

bool is_builtin_type(const std::string &name)
{
  return (name == "int" || name == "float" || name == "bool");
}

bool is_consensus_type(const std::string &name)
{
  return (name == "uint64");
}

std::map<std::string, std::string> consensus_func_to_type = {
  {"hash", "uint256"}};

bool is_consensus_func(const std::string &name)
{
  return consensus_func_to_type.find(name) != consensus_func_to_type.end();
}

std::string get_type_from_consensus_func(const std::string &name)
{
  if (!is_consensus_func(name))
    return std::string();

  return consensus_func_to_type[name];
}
