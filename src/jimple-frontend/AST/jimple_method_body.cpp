//
// Created by rafaelsamenezes on 21/09/2021.
//

#include <jimple-frontend/AST/jimple_method_body.h>
#include <jimple-frontend/AST/jimple_declaration.h>
#include <jimple-frontend/AST/jimple_statement.h>
#include <memory>
#include <util/std_code.h>
#include <util/expr_util.h>
#include <util/message/format.h>

exprt jimple_full_method_body::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  code_blockt block;
  for(auto const &stmt : this->members)
    block.operands().push_back(stmt->to_exprt(ctx, class_name, function_name));

  return block;
}

void jimple_full_method_body::from_json(const json &j)
{
  std::shared_ptr<jimple_label> label;
  for(auto &x : j)
  {
    std::shared_ptr<jimple_method_field> to_add;

    auto stmt = x.at("object").get<std::string>();
    // I think that this is the best way without
    // adding some crazy function pointer.
    switch(from_map[stmt])
    {
    case statement::Declaration:
    {
      jimple_declaration d;
      x.get_to(d);
      to_add = std::make_shared<jimple_declaration>(d);
      break;
    }
    case statement::Identity:
    {
      jimple_identity s;
      x.at("statement").at("identity").get_to(s);
      to_add = std::make_shared<jimple_identity>(s);
      break;
    }
    case statement::StaticInvoke:
    {
      jimple_invoke s;
      x.get_to(s);
      to_add = std::make_shared<jimple_invoke>(s);
      break;
    }
    case statement::Return:
    {
      jimple_return s;
      x.at("statement").at("return").get_to(s);
      to_add = std::make_shared<jimple_return>(s);
      break;
    }
    case statement::Label:
    {
      jimple_label s;
      x.at("label").get_to(s);
      if(label)
        members.push_back(std::move(label));
      label = std::make_shared<jimple_label>(s);
      continue;
    }
    case statement::Goto:
    {
      jimple_goto s;
      x.at("goto").get_to(s);
      to_add = std::make_shared<jimple_goto>(s);
      break;
    }
    case statement::Assignment:
    {
      jimple_assignment s;
      x.get_to(s);
      to_add = std::make_shared<jimple_assignment>(s);
      break;
    }
    case statement::Assertion:
    {
      jimple_assertion s;
      x.get_to(s);
      to_add = std::make_shared<jimple_assertion>(s);
      break;
    }
    case statement::If:
    {
      jimple_if s;
      x.get_to(s);
      to_add = std::make_shared<jimple_if>(s);
      break;
    }
    default:
      throw fmt::format("Unknown type {}", stmt);
    }

    if(label)
      label->push_into_label(to_add);
    else
      members.push_back(std::move(to_add));
  }
  if(label)
    members.push_back(std::move(label));
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
