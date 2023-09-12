#include "python-frontend/python_converter.h"
#include "util/std_code.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

const char* json_filename = "/tmp/ast.json";

python_converter::python_converter(contextt &_context) : context(_context)
{
}

bool python_converter::convert()
{
  std::ifstream f(json_filename);
  json ast = json::parse(f);

  for(auto &element : ast["body"])
  {
    if(element["_type"] == "Assign")
    {
      std::string lhs("");

      for(const auto &target : element["targets"])
      {
        if(target["_type"] == "Name")
        {
          lhs = target["id"];
        }
      }

      std::cout << lhs << " = ";

      json value = element["value"];

      if(value["_type"] == "BinOp")
      {
        json left = value["left"];
        json op = value["op"];
        json right = value["right"];
        std::cout << left["value"] << op["_type"] << right["value"] << std::endl;

        exprt expr = code_skipt();

      }
    }
  }

  return false;
}
