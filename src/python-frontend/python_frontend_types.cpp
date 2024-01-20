#include <python_frontend_types.h>

bool is_builtin_type(const std::string &name)
{
  return (name == "int" || name == "float" || name == "bool");
}
