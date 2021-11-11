//
// Created by rafaelsamenezes on 21/09/2021.
//

#ifndef ESBMC_JIMPLE_METHOD_BODY_H
#define ESBMC_JIMPLE_METHOD_BODY_H

#include <jimple-frontend/AST/jimple_ast.h>
#include <util/std_code.h>

/**
 * @brief A Jimple method definition
 * 
 * Something such as: public void foo() { }
 */
class jimple_method_body : public jimple_ast
{
public:
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const
  {
    exprt dummy;
    return dummy;
  }
};

/**
 * @brief The contents of a Jimple method
 *
 * This can be statements or declarations
 */
class jimple_method_field : public jimple_ast
{
public:
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const
  {
    dump();
    code_skipt dummy;
    return dummy;
  }
};

/**
 * @brief A Jimple method definition
 * 
 * Something such as: public void foo();
 */
class jimple_empty_method_body : public jimple_method_body
{
};

/**
 * @brief A Jimple method definition
 * 
 * Something such as: public void foo() { }
 */
class jimple_full_method_body : public jimple_method_body
{
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  enum class statement
  {
    Assertion, // TODO: This will be removed eventually
    Assignment,
    Identity,
    StaticInvoke,
    Return,
    Label,
    Goto,
    If,
    Declaration,
    Throw
  };

protected:
  std::vector<std::shared_ptr<jimple_method_field>> members;

private:
  std::map<std::string, statement> from_map = {
    {"Variable", statement::Declaration},
    {"identity", statement::Identity},
    {"StaticInvoke", statement::StaticInvoke},
    {"return", statement::Return},
    {"label", statement::Label},
    {"goto", statement::Goto},
    {"SetVariable", statement::Assignment},
    {"Assert", statement::Assertion},
    {"if", statement::If},
    {"throw", statement::Throw}};

  std::map<statement, std::string> to_map = {
    {statement::Identity, "Identity"},
    {statement::StaticInvoke, "StaticInvoke"},
    {statement::Return, "Return"},
    {statement::Label, "Label"},
    {statement::Goto, "Goto"},
    {statement::Assignment, "Assignment"},
    {statement::Assertion, "Assertion"},
    {statement::If, "If"},
    {statement::Declaration, "Declaration"},
    {statement::Throw, "Throw"}};
};

#endif //ESBMC_JIMPLE_METHOD_BODY_H
