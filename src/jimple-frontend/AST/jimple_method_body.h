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
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  enum class statement
  {
    Assertion,       // TODO: This will be removed eventually (for debugging)
    Assignment,      // A = 42
    AssignmentDeref, // A[2] = 3
    Identity, // @this, @parameter0, @parameter1, ...; This will be removed as it can solved directly in the frontend
    StaticInvoke,  // foo() (where foo is a static function)
    SpecialInvoke, // A.foo() (where A is an object)
    Return,        // return; return 42;
    Label,         // 1:, 2:; (GOTO labels)
    Goto,          // goto 1;
    If,            // if <expr> goto <Label>
    Declaration,   // int a;
    Throw          // throw <expr>
  };

  std::vector<std::shared_ptr<jimple_method_field>> members;

private:
  std::map<std::string, statement> from_map = {
    {"Variable", statement::Declaration},
    {"identity", statement::Identity},
    {"StaticInvoke", statement::StaticInvoke},
    {"SpecialInvoke", statement::SpecialInvoke},
    {"Return", statement::Return},
    {"Label", statement::Label},
    {"Goto", statement::Goto},
    {"SetVariable", statement::Assignment},
    {"SetVariableDeref", statement::AssignmentDeref},
    {"Assert", statement::Assertion},
    {"If", statement::If},
    {"Throw", statement::Throw}};

  std::map<statement, std::string> to_map = {
    {statement::Identity, "Identity"},
    {statement::StaticInvoke, "StaticInvoke"},
    {statement::SpecialInvoke, "SpecialInvoke"},
    {statement::Return, "Return"},
    {statement::Label, "Label"},
    {statement::Goto, "Goto"},
    {statement::Assignment, "Assignment"},
    {statement::AssignmentDeref, "AssignmentDeref"},
    {statement::Assertion, "Assertion"},
    {statement::If, "If"},
    {statement::Declaration, "Declaration"},
    {statement::Throw, "Throw"}};
};

#endif //ESBMC_JIMPLE_METHOD_BODY_H
