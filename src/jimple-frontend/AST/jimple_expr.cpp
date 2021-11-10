#include <jimple-frontend/AST/jimple_expr.h>
#include <util/std_code.h>
#include <util/expr_util.h>

void jimple_constant::from_json(const json &j)
{
  j.at("value").get_to(value);
}

exprt jimple_constant::to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const
{
  auto as_number = std::stoi(value);
  return constant_exprt(
    integer2binary(as_number, 10), integer2string(as_number), int_type());
};

void jimple_symbol::from_json(const json &j)
{
  j.at("value").get_to(var_name);
}

exprt jimple_symbol::to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const
{
  std::ostringstream oss;
    oss << class_name << ":" << function_name << "@" << var_name;

    symbolt &test = *ctx.find_symbol(oss.str());
    return symbol_expr(test);
};

void jimple_add::from_json(const json &j)
{
  j.at("value").get_to(value);
  j.at("var_name").get_to(var_name);
}

exprt jimple_add::to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const
{
  std::ostringstream oss;
    oss << class_name << ":" << function_name << "@" << var_name;

    symbolt &test = *ctx.find_symbol(oss.str());
    auto lhs = symbol_expr(test);

    auto as_number = std::stoi(value);
    auto rhs = constant_exprt(
    integer2binary(as_number, 10), integer2string(as_number), int_type());


    return gen_binary("add", int_type(), lhs, rhs);
};