//
// Created by rafaelsamenezes on 21/09/2021.
//

#ifndef ESBMC_JIMPLE_TYPE_H
#define ESBMC_JIMPLE_TYPE_H

#include <jimple-frontend/AST/jimple_ast.h>

class jimple_type : public jimple_ast {
};

class jimple_void_type : public jimple_type {
public:
  virtual void from_json(const json& j) override;
  virtual std::string to_string() override;
};
class jimple_nonvoid_type : public jimple_type {
};

class jimple_base_type : public jimple_nonvoid_type {};
class jimple_object_type : public jimple_nonvoid_type {};
#endif //ESBMC_JIMPLE_TYPE_H
