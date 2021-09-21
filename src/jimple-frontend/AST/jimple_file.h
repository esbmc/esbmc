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

class jimple_file : public jimple_ast {
public:
  enum class file_type {
    Class,
    Interface
  };

  file_type from_string(const std::string &name);
  std::string to_string(const file_type &ft);
  virtual void from_json(const json& j) override;
  virtual std::string to_string() override;
protected:
  file_type mode;
  std::string class_name;
  std::string extends;
  std::string implements;
  jimple_modifiers m;
  std::vector<jimple_class_method> body;
};


#endif //ESBMC_JIMPLE_FILE_H
