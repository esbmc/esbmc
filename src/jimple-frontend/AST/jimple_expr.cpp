#include <jimple-frontend/AST/jimple_expr.h>
#include <util/arith_tools.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/std_code.h>
#include <util/std_expr.h>

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

  if(expr_type == "string_constant")
  {
    jimple_constant c;
    return std::make_shared<jimple_constant>(c);
  }

  if(expr_type == "symbol")
  {
    jimple_symbol c;
    c.from_json(j);
    return std::make_shared<jimple_symbol>(c);
  }

  if(expr_type == "static_invoke")
  {
    jimple_expr_invoke c;
    c.from_json(j);
    return std::make_shared<jimple_expr_invoke>(c);
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

  if(expr_type == "new")
  {
    jimple_new c;
    c.from_json(j);
    return std::make_shared<jimple_new>(c);
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

  if(expr_type == "field_access")
  {
    jimple_field_access c;
    c.from_json(j);
    return std::make_shared<jimple_field_access>(c);
  }

  throw "Invalid expr type";
}

void jimple_binop::from_json(const json &j)
{
  j.at("operator").get_to(binop);
  // TODO, make hashmap for each operator
  if(binop == "==")
    binop = "=";
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
  if(t.is_array())
    return to_array_type(t).size();
  return constant_exprt(integer2binary(0, 10), integer2string(0), int_type());
};

void jimple_newarray::from_json(const json &j)
{
  size = get_expression(j.at("size"));
  jimple_type t;
  j.at("type").get_to(t);
  type = std::make_shared<jimple_type>(t);
}

void jimple_new::from_json(const json &j)
{
  size = std::make_shared<jimple_constant>("1");
  jimple_type t;
  j.at("type").get_to(t);
  type = std::make_shared<jimple_type>(t);
}

void jimple_expr_invoke::from_json(const json &j)
{
  lhs = nil_exprt();
  j.at("base_class").get_to(base_class);
  j.at("method").get_to(method);
  for(auto x : j.at("parameters"))
  {
    parameters.push_back(std::move(jimple_expr::get_expression(x)));
  }
  method += "_" + get_hash_name();
}

exprt jimple_expr_invoke::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // TODO: Move intrinsics to backend
  if(base_class == "kotlin.jvm.internal.Intrinsics")
  {
    code_skipt skip;
    return skip;
  }

  code_blockt block;
  code_function_callt call;

  std::ostringstream oss;
  oss << base_class << ":" << method;

  auto symbol = ctx.find_symbol(oss.str());
  call.function() = symbol_expr(*symbol);
  if(!lhs.is_nil())
    call.lhs() = lhs;

  for(auto i = 0; i < parameters.size(); i++)
  {
    // Just adding the arguments should be enough to set the parameters
    auto parameter_expr =
      parameters[i]->to_exprt(ctx, class_name, function_name);
    call.arguments().push_back(parameter_expr);
    // Hack, manually adding parameters
    std::ostringstream oss;
    oss << "@parameter" << i;
    auto temp = get_symbol_name(base_class, method, oss.str());
    symbolt &added_symbol = *ctx.find_symbol(temp);
    code_assignt assign(symbol_expr(added_symbol), parameter_expr);
    block.operands().push_back(assign);
  }
  block.operands().push_back(call);
  /*
   // Create a sideffect call to represent the allocation
  side_effect_expr_function_callt sideeffect;
  sideeffect.function() = call.function();
  sideeffect.arguments() = call.arguments();
  sideeffect.location() = call.location();
  sideeffect.type() =
    static_cast<const typet &>(call.function().type().return_type());
  
  block.operands().push_back(sideeffect); */
  return block;
}

exprt jimple_newarray::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // Generate base_type
  auto base_type = type->to_typet();

  // Create tmp var to receive the allocation
  auto tmp_symbol =
    get_temp_symbol(pointer_typet(base_type), class_name, function_name);
  symbolt &tmp_added_symbol = *ctx.move_symbol_to_context(tmp_symbol);

  // Create a function call for allocation
  code_function_callt call;
  auto alloca_symbol = get_allocation_function();
  symbolt &added_symbol = *ctx.move_symbol_to_context(alloca_symbol);
  call.function() = symbol_expr(added_symbol);
  call.function().type() = pointer_typet(empty_typet());

  // LHS of call is the tmp var
  call.lhs() = symbol_expr(tmp_added_symbol);
  auto as_number = std::stoi(base_type.width().as_string()) / 8;
  auto value_operand = gen_binary(
    "*",
    uint_type(),
    size->to_exprt(ctx, class_name, function_name),
    constant_exprt(
      integer2binary(as_number, 10), integer2string(as_number), int_type()));

  // Define the base type to be used as primitive in allocation
  //value_operand.set("#c_sizeof_type", base_type);
  call.arguments().push_back(value_operand);

  // Create a sideffect call to represent the allocation
  side_effect_expr_function_callt sideeffect;
  sideeffect.function() = call.function();
  sideeffect.arguments() = call.arguments();
  sideeffect.location() = call.location();
  sideeffect.type() =
    static_cast<const typet &>(call.function().type().return_type());
  return sideeffect;
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
  auto i = index->to_exprt(ctx, class_name, function_name);
  auto index = index_exprt(arr, i, arr.type().subtype());
  exprt &array_expr = index.op0();
  exprt &index_expr = index.op1();

  exprt addition("+", array_expr.type());
  addition.operands().swap(index.operands());

  index.move_to_operands(addition);
  index.id("dereference");
  index.type() = array_expr.type().subtype();

  return index;
};

exprt jimple_nondet::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  exprt nondet_expr("nondet", int_type());
  return nondet_expr;
};

void jimple_field_access::from_json(const json &j)
{
  j.at("from").get_to(from);
  j.at("field").get_to(field);
  jimple_type t;
  j.at("type").get_to(t);
  type = std::make_shared<jimple_type>(t);
}

exprt jimple_field_access::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  auto result = gen_zero(type->to_typet());
  // HACK: For now I will set some intrinsics directly (this should go to SYMEX)
  if(from == "kotlin._Assertions" && field == "ENABLED")
    result.make_true();
  // TODO: Needs OOP members
  return result;
};