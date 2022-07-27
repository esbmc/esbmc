#ifndef ESBMC_JIMPLE_METHOD_BODY_H
#define ESBMC_JIMPLE_METHOD_BODY_H

#include <jimple-frontend/AST/jimple_ast.h>
#include <util/std_code.h>

/**
 * @brief A Jimple method declaration
 *
 * Something such as: public void foo() { }
 */
class jimple_method_body : public jimple_ast
{
public:
  virtual exprt
  to_exprt(contextt &, const std::string &, const std::string &) const
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
  virtual exprt
  to_exprt(contextt &, const std::string &, const std::string &) const
  {
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
    Assignment, // A = 42
    Identity, // @this, @parameter0, @parameter1, ...; This will be removed as it can solved directly in the frontend
    StaticInvoke, // foo() (where foo is a static function)
    SpecialInvoke, // Special methods of the class: constructors/static-constructor
    VirtualInvoke, // A.foo() (where A is an object)
    Return,        // return; return 42;
    Label,         // 1:, 2:; (GOTO labels)
    Goto,          // goto 1;
    If,            // if <expr> goto <Label>
    Declaration,   // int a;
    Throw,         // throw <expr>
    Location       // Extra, reffers to the line number
  };

  std::vector<std::shared_ptr<jimple_method_field>> members;

private:
  std::map<std::string, statement> from_map = {
    {"Variable", statement::Declaration},
    {"identity", statement::Identity},
    {"StaticInvoke", statement::StaticInvoke},
    {"SpecialInvoke", statement::SpecialInvoke},
    {"VirtualInvoke", statement::VirtualInvoke},
    {"Return", statement::Return},
    {"Label", statement::Label},
    {"Goto", statement::Goto},
    {"SetVariable", statement::Assignment},
    {"If", statement::If},
    {"Throw", statement::Throw},
    {"Location", statement::Location}};

  std::map<statement, std::string> to_map = {
    {statement::Identity, "Identity"},
    {statement::StaticInvoke, "StaticInvoke"},
    {statement::SpecialInvoke, "SpecialInvoke"},
    {statement::VirtualInvoke, "VirtualInvoke"},
    {statement::Return, "Return"},
    {statement::Label, "Label"},
    {statement::Goto, "Goto"},
    {statement::Assignment, "Assignment"},
    {statement::If, "If"},
    {statement::Declaration, "Declaration"},
    {statement::Throw, "Throw"},
    {statement::Location, "Location"}};
};

#endif //ESBMC_JIMPLE_METHOD_BODY_H
