#include "python-frontend/python_converter.h"
#include "util/std_code.h"
#include "util/c_types.h"
#include "util/arith_tools.h"
#include "util/expr_util.h"
#include "util/message.h"

#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

const char *json_filename = "/tmp/ast.json";

python_converter::python_converter(contextt &_context) : context(_context)
{
}

std::string get_op(const std::string &op)
{
  if(op == "Div")
    return "/";
  if(op == "Add")
    return "+";
  return std::string();
}

bool python_converter::convert()
{
  codet init_code = code_blockt();
  init_code.make_block();

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

      typet type = uint_type();
      locationt location;
      location.set_line(1);
      location.set_file("program.py");
      symbolt symbol;
      symbol.mode = "Python";
      symbol.module = "program.py";
      symbol.location = location;
      symbol.type = type;
      symbol.name = lhs;
      symbol.id = "c:@" + lhs;
      symbol.lvalue = true;
      symbol.static_lifetime = false;
      symbol.file_local = true;
      symbol.is_extern = false;

      json value = element["value"];
      if(value["_type"] == "BinOp")
      {
        int left = value["left"]["value"].get<int>();
        std::string op = get_op(value["op"]["_type"].get<std::string>());
        int right = value["right"]["value"].get<int>();

        exprt left_op = from_integer(left, int_type());
        exprt right_op = from_integer(right, int_type());

        exprt val(op, int_type());
        val.copy_to_operands(left_op, right_op);

        symbol.value = val;

        init_code.copy_to_operands(code_assignt(symbol_expr(symbol), val));
      }

      context.add(symbol);
    }
  }

  // add "main"
  symbolt main_symbol;

  code_typet main_type;
  main_type.return_type() = empty_typet();

  main_symbol.id = "__ESBMC_main";
  main_symbol.name = "__ESBMC_main";
  main_symbol.type.swap(main_type);
  main_symbol.value.swap(init_code);

  if(context.move(main_symbol))
  {
    log_error("main already defined by another language module");
    return true;
  }

  return false;
}
