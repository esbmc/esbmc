//
// Created by rafaelsamenezes on 22/09/2021.
//

#ifndef ESBMC_JIMPLE_STATEMENT_H
#define ESBMC_JIMPLE_STATEMENT_H

#include <jimple-frontend/AST/jimple_method_body.h>
#include <jimple-frontend/AST/jimple_type.h>
#include <jimple-frontend/AST/jimple_expr.h>

/**
 * @brief Base class for Jimple Statements
 *
 * They can represent a range of operations such as:
 * if/goto/assignments/etc...
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

  const std::string &getLocalName() const
  {
    return local_name;
  }

  const std::string &getAtIdentifier() const
  {
    return at_identifier;
  }

  const jimple_type &getT() const
  {
    return t;
  }

protected:
  std::string local_name;
  std::string at_identifier;
  jimple_type t;
};

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

class jimple_return : public jimple_statement
{
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;
};

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
