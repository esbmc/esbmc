#include <jimple-frontend/AST/jimple_ast.h>
#include <jimple-frontend/AST/jimple_type.h>

#pragma once
class jimple_expr : public jimple_ast
{
public:
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const
  {
    exprt val("at_identifier");
    return val;
  };

  // Expressions parsing can be recursive
  static std::shared_ptr<jimple_expr> get_expression(const json &j);
};

class jimple_constant : public jimple_expr
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return value;
  }
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

protected:
  std::string value;
};

class jimple_symbol : public jimple_expr
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return var_name;
  }
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

protected:
  std::string var_name;
};

class jimple_binop : public jimple_expr
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return "Jimple BinOP\n";
  }

  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

protected:
  std::string binop;
  std::shared_ptr<jimple_expr> lhs;
  std::shared_ptr<jimple_expr> rhs;
};
