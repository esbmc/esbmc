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
