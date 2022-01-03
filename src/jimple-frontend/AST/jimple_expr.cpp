#include <jimple-frontend/AST/jimple_expr.h>
#include <util/std_code.h>
#include <util/expr_util.h>
#include <util/c_typecast.h>
#include <util/c_types.h>

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
  // 1. Look over the local scope
  auto symbol_name = get_symbol_name(class_name, function_name, var_name);
  symbolt &s = *ctx.find_symbol(symbol_name);

  // TODO:
  // 2. Look over the class scope
  // 3. Look over the global scope (possibly don't need)

  return symbol_expr(s);
};

std::shared_ptr<jimple_expr> jimple_expr::get_expression(const json &j)
{
  std::string expr_type;
  if(!j.contains("expr_type"))
  {
    jimple_constant c;
    c.setValue("0");
    return std::make_shared<jimple_constant>(c);
  }

  j.at("expr_type").get_to(expr_type);

  // TODO: hashmap, the standard is not stable enough yet
  if(expr_type == "constant")
  {
    jimple_constant c;
    c.from_json(j);
    return std::make_shared<jimple_constant>(c);
  }

  if(expr_type == "symbol")
  {
    jimple_symbol c;
    c.from_json(j);
    return std::make_shared<jimple_symbol>(c);
  }

  if(expr_type == "binop")
  {
    jimple_binop c;
    c.from_json(j);
    return std::make_shared<jimple_binop>(c);
  }

  if(expr_type == "cast")
  {
    jimple_cast c;
    c.from_json(j);
    return std::make_shared<jimple_cast>(c);
  }

  if(expr_type == "lengthof")
  {
    jimple_cast c;
    c.from_json(j);
    return std::make_shared<jimple_cast>(c);
  }

  if(expr_type == "newarray")
  {
    jimple_newarray c;
    c.from_json(j);
    return std::make_shared<jimple_newarray>(c);
  }

  if(expr_type == "deref")
  {
    jimple_deref c;
    c.from_json(j);
    return std::make_shared<jimple_deref>(c);
  }

  if(expr_type == "nondet")
  {
    jimple_nondet c;
    return std::make_shared<jimple_nondet>(c);
  }

  throw "Invalid expr type";
}

void jimple_binop::from_json(const json &j)
{
  j.at("operator").get_to(binop);
  lhs = get_expression(j.at("lhs"));
  rhs = get_expression(j.at("rhs"));
}

exprt jimple_binop::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  auto lhs_expr = lhs->to_exprt(ctx, class_name, function_name);
  return gen_binary(
    binop,
    lhs_expr.type(),
    lhs_expr,
    rhs->to_exprt(ctx, class_name, function_name));
};

void jimple_cast::from_json(const json &j)
{
  jimple_type t;
  j.at("to").get_to(t);
  to = std::make_shared<jimple_type>(t);
  from = get_expression(j.at("from"));
  
}

exprt jimple_cast::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  auto from_expr = from->to_exprt(ctx, class_name, function_name);
  c_typecastt c_typecast(ctx);
  
  c_typecast.implicit_typecast(from_expr, to->to_typet());
  return from_expr;
};

void jimple_lenghof::from_json(const json &j)
{
  from = get_expression(j.at("from"));
}

exprt jimple_lenghof::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  auto t = from->to_exprt(ctx, class_name, function_name).type();
  if(t.is_array()) return to_array_type(t).size();
  return constant_exprt(
    integer2binary(0, 10), integer2string(0), int_type());
};

void jimple_newarray::from_json(const json &j)
{
  size = get_expression(j.at("size"));
  jimple_type t;
  j.at("type").get_to(t);
  type = std::make_shared<jimple_type>(t);
}

#include <util/std_expr.h>
exprt jimple_newarray::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
    return gen_zero(array_typet(
      type->to_typet(),
      size->to_exprt(ctx,class_name,function_name)));
};


void jimple_deref::from_json(const json &j)
{
  base = get_expression(j.at("base"));
  index = get_expression(j.at("index"));
}

exprt jimple_deref::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  auto arr = base->to_exprt(ctx, class_name, function_name);
  auto i = index->to_exprt(ctx,class_name,function_name);
  return index_exprt(arr, i, arr.type());
};

exprt jimple_nondet::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  exprt nondet_expr("nondet", int_type());
  return nondet_expr;
};