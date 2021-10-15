//
// Created by rafaelsamenezes on 21/09/2021.
//

#include <jimple-frontend/AST/jimple_method_body.h>
#include <jimple-frontend/AST/jimple_declaration.h>
#include <jimple-frontend/AST/jimple_statement.h>
#include <util/std_code.h>
#include <util/expr_util.h>

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
  for(auto &x : j)
  {
    /* NOTE: Since its only one condition I don't think
     * that a HashMap is needed */

    // Declaration Parsing
    if(x.contains("declaration"))
    {
      jimple_declaration d;
      x.at("declaration").get_to(d);
      members.push_back(std::make_shared<jimple_declaration>(d));
    }
    // Statement Parsing
    else
    {
      // TODO: Remove this and add a HashMap
      auto stmt = x.at("statement").at("stmt").get<std::string>();
      if(stmt == "identity")
      {
        jimple_identity s;
        x.at("statement").at("identity").get_to(s);
        members.push_back(std::make_shared<jimple_identity>(s));
      }
      else if(stmt == "invoke")
      {
        jimple_invoke s;
        x.at("statement").get_to(s);
        members.push_back(std::make_shared<jimple_invoke>(s));
      }
      else if(stmt == "return")
      {
        jimple_return s;
        x.at("statement").at("return").get_to(s);
        members.push_back(std::make_shared<jimple_return>(s));
      }
      else if(stmt == "label")
      {
        jimple_label s;
        x.at("statement").at("label").get_to(s);
        members.push_back(std::make_shared<jimple_label>(s));
      }
      else if(stmt == "assignment")
      {
        jimple_assignment s;
        x.at("statement").at("assignment").get_to(s);
        members.push_back(std::make_shared<jimple_assignment>(s));
      }
      else if(stmt == "assertion")
      {
        jimple_assertion s;
        x.at("statement").at("assertion").get_to(s);
        members.push_back(std::make_shared<jimple_assertion>(s));
      }
    }
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
