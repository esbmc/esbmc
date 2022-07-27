#include <jimple-frontend/AST/jimple_method_body.h>
#include <jimple-frontend/AST/jimple_declaration.h>
#include <jimple-frontend/AST/jimple_statement.h>
#include <memory>
#include <util/std_code.h>
#include <util/expr_util.h>

exprt jimple_full_method_body::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  code_blockt block;
  for(auto const &stmt : this->members)
  {
    auto expression = stmt->to_exprt(ctx, class_name, function_name);
    auto l = jimple_ast::get_location(class_name, function_name);
    if(stmt->line_location != -1)
    {
      l.set_line(stmt->line_location);
    }
    expression.location() = l;
    block.operands().push_back(expression);
  }

  return block;
}

void jimple_full_method_body::from_json(const json &stmts)
{
  int inner_location = -1;
  for(auto &stmt : stmts)
  {
    std::shared_ptr<jimple_method_field> to_add;

    auto mode = stmt.at("object").get<std::string>();
    // I think that this is the best way without
    // adding some crazy function pointer.
    switch(from_map[mode])
    {
    case statement::Declaration:
    {
      jimple_declaration d;
      stmt.get_to(d);
      to_add = std::make_shared<jimple_declaration>(d);
      break;
    }
    case statement::Location:
    {
      std::string location_str;
      stmt.at("line").get_to(location_str);
      inner_location = std::stoi(location_str);
      //log_debug("Setting location as: {}", inner_location);
      continue;
    }
    case statement::Identity:
    {
      jimple_identity s;
      stmt.at("statement").at("identity").get_to(s);
      to_add = std::make_shared<jimple_identity>(s);
      break;
    }
    case statement::StaticInvoke:
    {
      jimple_invoke s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_invoke>(s);
      break;
    }
    case statement::VirtualInvoke:
    {
      jimple_invoke s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_invoke>(s);
      break;
    }
    case statement::SpecialInvoke:
    {
      jimple_invoke s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_invoke>(s);
      break;
    }
    case statement::Return:
    {
      jimple_return s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_return>(s);
      break;
    }
    case statement::Label:
    {
      jimple_label s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_label>(s);
      break;
    }
    case statement::Goto:
    {
      jimple_goto s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_goto>(s);
      break;
    }
    case statement::Assignment:
    {
      jimple_assignment s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_assignment>(s);
      break;
    }
    case statement::Throw:
    {
      jimple_throw s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_throw>(s);
      break;
    }
    case statement::If:
    {
      jimple_if s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_if>(s);
      break;
    }
    default:
      log_error("Unknown type {}", stmt);
      abort();
    }

    to_add->line_location = inner_location;
    members.push_back(std::move(to_add));
  }
}
std::string jimple_full_method_body::to_string() const
{
  std::ostringstream oss;
  for(auto &x : members)
  {
    oss << "\n\t\t" << x->to_string();
  }
  return oss.str();
}
