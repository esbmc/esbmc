/*******************************************************************\
Module: Jimple File AST
Author: Rafael Sá Menezes
Date: September 2021
Description: Jimple File AST parser and holder
\*******************************************************************/

#ifndef ESBMC_JIMPLE_FILE_H
#define ESBMC_JIMPLE_FILE_H

#include <jimple-frontend/AST/jimple_ast.h>
#include <jimple-frontend/AST/jimple_modifiers.h>
#include <jimple-frontend/AST/jimple_class_member.h>
#include <util/expr.h>

class jimple_file : public jimple_ast {
public:
  jimple_file() = default;
  void load_file(const std::string& path);

  enum class file_type {
    Class,
    Interface
  };
  virtual exprt to_exprt(const messaget &msg, contextt &ctx) const override;
  file_type from_string(const std::string &name);
  std::string to_string(const file_type &ft) const;
  virtual void from_json(const json& j) override;
  virtual std::string to_string() const override;

protected:
public:
  file_type getMode() const
  {
    return mode;
  }
  const std::string &getClassName() const
  {
    return class_name;
  }
  const std::string &getExtends() const
  {
    return extends;
  }
  const std::string &getImplements() const
  {
    return implements;
  }
  const jimple_modifiers &getM() const
  {
    return m;
  }
  const std::vector<jimple_class_method> &getBody() const
  {
    return body;
  }

protected:
  file_type mode;
  std::string class_name;
  std::string extends;
  std::string implements;
  jimple_modifiers m;
  std::vector<jimple_class_method> body;
};


#endif //ESBMC_JIMPLE_FILE_H
