/*******************************************************************\
Module: Jimple Statements AST
Author: Rafael Sá Menezes
Date: September 2021
Description: Jimple Statements definitions
  They can represent a range of operations such as:
  if/goto/assignments/etc..
\*******************************************************************/

#ifndef ESBMC_JIMPLE_STATEMENT_H
#define ESBMC_JIMPLE_STATEMENT_H

#include <jimple-frontend/AST/jimple_method_body.h>
#include <jimple-frontend/AST/jimple_type.h>
#include <jimple-frontend/AST/jimple_expr.h>

/**
 * @brief Base class for Jimple Statements.
 */
class jimple_statement : public jimple_method_field
{
};

// THIS IS STILL A TODO FROM THE STANDARD
class jimple_identity : public jimple_statement
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  const std::string &get_local_name() const
  {
    return local_name;
  }

  const std::string &get_at_identifier() const
  {
    return at_identifier;
  }

  const jimple_type &get_type() const
  {
    return type;
  }

protected:
  std::string local_name;
  std::string at_identifier;
  jimple_type type;
};

/**
 * @brief A function call
 * 
 * foo(42);
 */
class jimple_invoke : public jimple_statement
{
public:
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  const std::string &getBaseClass() const
  {
    return base_class;
  }

  const std::string &getMethod() const
  {
    return method;
  }

  const std::vector<std::shared_ptr<jimple_expr>> &getParameters() const
  {
    return parameters;
  }

protected:
  // We need an unique name for each function
  std::string get_hash_name() const
  {
    // TODO: use some hashing to also use the types
    // TODO: DRY
    return std::to_string(parameters.size());
  }
  std::string base_class;
  std::string method;
  std::vector<std::shared_ptr<jimple_expr>> parameters;
};

/**
 * @brief Return statement
 * 
 * return 42;
 */
class jimple_return : public jimple_statement
{
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;
  std::shared_ptr<jimple_expr> expr;
};

/**
 * @brief A GOTO label
 * 
 * label1:
 *    ...
 */
class jimple_label : public jimple_statement
{
public:
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  const std::string &getLabel() const
  {
    return label;
  }

protected:
  std::string label;
  std::shared_ptr<jimple_full_method_body> members;
};

/**
 * @brief Goto statement
 * 
 * goto label1;
 */
class jimple_goto : public jimple_statement
{
public:
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  const std::string &getLabel() const
  {
    return label;
  }

protected:
  std::string label;
};

/**
 * @brief An assignment statement
 * 
 * a = 42;
 */
class jimple_assignment : public jimple_statement
{
public:
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;

  const std::string &getVariable() const
  {
    return variable;
  }

  const std::shared_ptr<jimple_expr> &getExpr() const
  {
    return expr;
  }

protected:
  std::string variable;
  std::shared_ptr<jimple_expr> expr;
  bool is_skip = false;
};

/**
 * @brief Assignment of a position of a vector
 * 
 * a[2] = 42;
 */
class jimple_assignment_deref : public jimple_assignment
{
public:
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;

  const std::shared_ptr<jimple_expr> &getPos() const
  {
    return pos;
  }

protected:
  std::shared_ptr<jimple_expr> pos;
};

// For debug
class jimple_assertion : public jimple_statement
{
public:
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;

  const std::string &getVariable() const
  {
    return variable;
  }
  const std::string &getValue() const
  {
    return value;
  }

protected:
  std::string variable;
  std::string value;
};

/**
 * @brief An IF statement
 * 
 * if 2 > 4 goto label3;
 */
class jimple_if : public jimple_statement
{
public:
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;

  const std::string &getLabel() const
  {
    return label;
  }

  std::shared_ptr<jimple_expr> &getCond()
  {
    return cond;
  }

protected:
  std::shared_ptr<jimple_expr> cond;
  std::string label;
};

/**
 * @brief A throw statement
 * 
 * throw 0;
 * 
 */
class jimple_throw : public jimple_statement
{
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;

protected:
  std::shared_ptr<jimple_expr> expr;
};

#endif //ESBMC_JIMPLE_STATEMENT_H
