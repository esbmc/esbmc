#include <python_module.h>
#include <unordered_map>

static const std::unordered_map<std::string, std::string> function_returns_map =
  {{"popen", "None"},
   {"listdir", "list"},
   {"makdirs", "None"},
   {"remove", "None"},
   {"exists", "bool"}};

python_module::python_module(const std::string &module_name)
  : module_name_(module_name)
{
}

bool python_module::is_standard_module() const
{
  return module_name_ == "os";
}

std::string
python_module::get_function_return(const std::string &function_name) const
{
  auto it = function_returns_map.find(function_name);
  if (it != function_returns_map.end())
  {
    return it->second;
  }
  return std::string();
}
