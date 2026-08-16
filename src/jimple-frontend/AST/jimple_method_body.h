#ifndef ESBMC_JIMPLE_METHOD_BODY_H
#define ESBMC_JIMPLE_METHOD_BODY_H

#include <jimple-frontend/AST/jimple_ast.h>
#include <util/irep/migrate.h>
#include <util/irep/std_code.h>

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

  /**
   * @brief The IREP2 form of the body, for symbolt::set_value(const expr2tc &).
   *
   * K.1 of docs/roadmap/scope-jimple-irep2.md. The default migrates whatever
   * to_exprt built, so a subclass that has not been converted yet keeps
   * working; each override that lands removes one migration rather than adding
   * one, which is why this migration runs from the seam downwards.
   */
  virtual expr2tc to_code2t(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const
  {
    expr2tc body;
    migrate_expr(to_exprt(ctx, class_name, function_name), body);
    return body;
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

  /**
   * @brief The IREP2 form of this statement, located at @p loc.
   *
   * K.2 of docs/roadmap/scope-jimple-irep2.md. The location is a parameter
   * rather than something the caller stamps afterwards: a code_*2t holds it in
   * a non-reflected field, so it has to be set while the node is still a legacy
   * exprt. As in jimple_method_body, the default migrates whatever to_exprt
   * built, so each override that lands removes a migration instead of adding
   * one.
   */
  virtual expr2tc to_code2t(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name,
    const locationt &loc) const
  {
    exprt e = to_exprt(ctx, class_name, function_name);
    // A nil location means "leave whatever to_exprt produced": jimple_label
    // does not stamp its members, where jimple_full_method_body does.
    if (!loc.is_nil())
      e.location() = loc;
    expr2tc stmt;
    migrate_expr(e, stmt);
    return stmt;
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

  virtual expr2tc to_code2t(
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
