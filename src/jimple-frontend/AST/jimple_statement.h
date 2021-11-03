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
protected:
  std::shared_ptr<jimple_expr> get_expression(const json &j);
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

protected:
  std::string local_name;
  std::string at_identifier;
  jimple_type t;
};

// TODO: Fix the parser
class jimple_invoke : public jimple_statement
{
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
protected:
  std::string base_class;
  std::string method;
  std::string parameters;
};

// TODO: Add return statement
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

  virtual void push_into_label(std::shared_ptr<jimple_method_field> &f)
  {
    members.push_back(std::move(f));
  }

protected:
  std::string label;
  std::vector<std::shared_ptr<jimple_method_field>> members;
};

class jimple_goto : public jimple_statement
{
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

protected:
  std::string label;
};

class jimple_assignment : public jimple_statement
{
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;

protected:
  std::string variable;
  std::shared_ptr<jimple_expr> expr;
};

class jimple_assertion : public jimple_statement
{
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;

protected:
  std::string variable;
  std::string value;
};

class jimple_if : public jimple_statement
{
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
  virtual std::string to_string() const override;
  virtual void from_json(const json &j) override;

protected:
  std::string variable;
  std::string value;
  std::string label;
};

#endif //ESBMC_JIMPLE_STATEMENT_H
