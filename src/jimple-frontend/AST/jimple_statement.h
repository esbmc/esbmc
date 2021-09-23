//
// Created by rafaelsamenezes on 22/09/2021.
//

#ifndef ESBMC_JIMPLE_STATEMENT_H
#define ESBMC_JIMPLE_STATEMENT_H

#include <jimple-frontend/AST/jimple_method_body.h>
#include <jimple-frontend/AST/jimple_type.h>

class jimple_statement : public jimple_method_field {
};

class jimple_identity : public jimple_statement
{
public:
  virtual void from_json(const json& j) override;
  virtual std::string to_string() const override;
protected:
  std::string local_name;
  std::string at_identifier;
  jimple_type t;
};

// TODO: Fix the parser
class jimple_invoke : public jimple_statement
{
  virtual std::string to_string() const override;
  virtual void from_json(const json& j) override;
};

// TODO: Add return statement
class jimple_return : public jimple_statement
{
  virtual std::string to_string() const override;
  virtual void from_json(const json& j) override;
};

class jimple_label : public jimple_statement
{
  virtual std::string to_string() const override;
  virtual void from_json(const json& j) override;
protected:
  std::string label;
};

class jimple_assignment : public jimple_statement
{
  virtual std::string to_string() const override;
  virtual void from_json(const json& j) override;
protected:
  std::string variable;
};

#endif //ESBMC_JIMPLE_STATEMENT_H
