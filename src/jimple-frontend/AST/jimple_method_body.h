//
// Created by rafaelsamenezes on 21/09/2021.
//

#ifndef ESBMC_JIMPLE_METHOD_BODY_H
#define ESBMC_JIMPLE_METHOD_BODY_H

#include <jimple-frontend/AST/jimple_ast.h>
class jimple_method_body : public jimple_ast {};
class jimple_method_field : public jimple_ast {};

class jimple_empty_method_body : public jimple_method_body {};

class jimple_full_method_body : public jimple_method_body {
  virtual void from_json(const json& j) override;
  virtual std::string to_string() override;

protected:
  std::vector<std::shared_ptr<jimple_method_field>> members;

};

#endif //ESBMC_JIMPLE_METHOD_BODY_H
