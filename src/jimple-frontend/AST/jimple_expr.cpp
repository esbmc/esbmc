#include <jimple-frontend/AST/jimple_expr.h>
#include <util/std_code.h>
#include <util/expr_util.h>

void jimple_constant::from_json(const json &j)
{
  j.at("value").get_to(value);
}

exprt jimple_constant::to_exprt() const
{
  return constant_exprt(
        integer2binary(0,10),
        integer2string(0),
        int_type());
};