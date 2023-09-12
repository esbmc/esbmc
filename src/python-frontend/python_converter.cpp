#include "python-frontend/python_converter.h"

#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

python_converter::python_converter(contextt &_context) : context(_context)
{
}

bool python_converter::convert()
{
  std::ifstream f("/tmp/ast.json");
  json ast = json::parse(f);
  return false;
}
