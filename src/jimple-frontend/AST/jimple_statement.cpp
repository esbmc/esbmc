//
// Created by rafaelsamenezes on 22/09/2021.
//

#include <util/std_code.h>
#include <util/std_expr.h>
#include <util/std_types.h>
#include <jimple-frontend/AST/jimple_statement.h>
#include <util/arith_tools.h>
void jimple_identity::from_json(const json &j)
{
  j.at("identifier").get_to(at_identifier);
  j.at("name").get_to(local_name);
  j.at("type").get_to(t);
}

exprt jimple_identity::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // TODO: Symbol-table / Typecast
  exprt val("at_identifier");
  symbolt &added_symbol = *ctx.find_symbol(local_name);
  symbolt rhs;
  rhs.name = "@" + at_identifier;
  rhs.id = "@" + at_identifier;
  code_assignt assign(symbol_expr(added_symbol), symbol_expr(rhs));
  return assign;
}
std::string jimple_identity::to_string() const
{
  std::ostringstream oss;
  oss << "Identity:  " << this->local_name << " = @" << at_identifier << " | "
      << t.to_string();
  return oss.str();
}

exprt jimple_return::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // TODO: jimple return with support to other returns
  typet return_type = empty_typet();
  code_returnt ret_expr;
  // TODO: jimple return should support values
  return ret_expr;
}

std::string jimple_return::to_string() const
{
  return "Return: (Nothing)";
}
void jimple_return::from_json(const json &j)
{
  // TODO
}
std::string jimple_label::to_string() const
{
  std::ostringstream oss;
  oss << "Label: " << this->label;
  return oss.str();
}

exprt jimple_label::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  exprt skip = code_skipt();
  // TODO: DRY (clang-c-converter)
  code_labelt c_label;
  c_label.set_label(label);

  code_blockt block;
  for(auto x : members)
  {
    block.operands().push_back(x->to_exprt(ctx, class_name, function_name));
  }
  block.dump();
  c_label.code() = to_code(block);

  //skip = c_label;
  return c_label;
}

void jimple_goto::from_json(const json &j)
{
  j.get_to(label);
}

std::string jimple_goto::to_string() const
{
  std::ostringstream oss;
  oss << "Goto: " << this->label;
  return oss.str();
}

exprt jimple_goto::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  code_gotot code_goto;
  code_goto.set_destination(label);
  return code_goto;
}

void jimple_label::from_json(const json &j)
{
  j.get_to(label);
}


std::string jimple_assignment::to_string() const
{
  std::ostringstream oss;
  oss << "Assignment: " << variable << " = " << expr->to_string();
  return oss.str();
}

void jimple_assignment::from_json(const json &j)
{
  j.at("name").get_to(variable);
  expr = get_expression(j.at("expression"));
}

exprt jimple_assignment::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  std::ostringstream oss;
  oss << class_name << ":" << function_name << "@" << variable;

  symbolt &added_symbol = *ctx.find_symbol(oss.str());
  code_assignt assign(symbol_expr(added_symbol), expr->to_exprt());
  return assign;
}

std::string jimple_if::to_string() const
{
  std::ostringstream oss;
  oss << "If: " << variable << " = " << value
      << " then goto " << label;
  return oss.str();
}

void jimple_if::from_json(const json &j)
{
  j.at("cond").at("equals").at("symbol").get_to(variable);
  j.at("cond").at("equals").at("value").get_to(value);
  j.at("goto").get_to(label);
}

exprt jimple_if::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
  {
    this->dump();
    std::ostringstream oss;
    oss << class_name << ":" << function_name << "@" << variable;

    symbolt &test = *ctx.find_symbol(oss.str());
    int as_number = std::stoi(value);
    exprt value_operand = from_integer(as_number, int_type());

    equality_exprt ge(symbol_expr(test), value_operand);

    code_gotot code_goto;
    code_goto.set_destination(label);

    //    exprt else_expr = code_skipt();

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(ge, code_goto);

    return if_expr;
  }


std::string jimple_assertion::to_string() const
{
  std::ostringstream oss;
  oss << "Assertion: " << variable << " = " << value;
  return oss.str();
}

void jimple_assertion::from_json(const json &j)
{
  j.at("equals").at("symbol").get_to(variable);
  j.at("equals").at("value").get_to(value);
  
  
}

exprt jimple_assertion::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  code_function_callt call;

  std::ostringstream oss;
  oss << class_name << ":" << function_name << "@" << variable;

  // TODO: move this from here
  std::string id, name;
  id = "__ESBMC_assert";
  name = "__ESBMC_assert";

  auto symbol =
    create_jimple_symbolt(code_typet(), class_name, name, id, function_name);

  symbolt &added_symbol = *ctx.move_symbol_to_context(symbol);

  call.function() = symbol_expr(added_symbol);

  symbolt &test = *ctx.find_symbol(oss.str());
  int as_number = std::stoi(value);
  exprt value_operand = from_integer(as_number, int_type());

  equality_exprt ge(symbol_expr(test), value_operand);
  not_exprt qwe(ge);
  call.arguments().push_back(qwe);

  array_of_exprt arr;
  // TODO: Create binop operation between symbol and value
  return call;
}

std::shared_ptr<jimple_expr> jimple_statement::get_expression(const json &j)
{
  std::string expr_type;
  j.at("expr_type").get_to(expr_type);

  // TODO: hashmap
  if(expr_type == "constant")
  {
    jimple_constant c;
    c.from_json(j);
    return std::make_shared<jimple_constant>(c);
  }

  jimple_constant d;
  return std::make_shared<jimple_constant>(d);
}



std::string jimple_invoke::to_string() const
{
  std::ostringstream oss;
  oss << "Invoke: " << method;
  return oss.str();
}

void jimple_invoke::from_json(const json &j)
{
  j.at("base_class").get_to(base_class);
  j.at("method").get_to(method);
  j.at("parameters").get_to(parameters);  
}

exprt jimple_invoke::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  code_function_callt call;

  std::ostringstream oss;
  oss << base_class << ":" << method;

  // TODO: move this from here
  std::string id, name;
  id = "__ESBMC_assert";
  name = "__ESBMC_assert";

  auto symbol = ctx.find_symbol(oss.str());  
  call.function() = symbol_expr(*symbol);

  array_of_exprt arr;
  // TODO: Create binop operation between symbol and value
  return call;
}
