//
// Created by rafaelsamenezes on 21/09/2021.
//

#ifndef ESBMC_JIMPLE_TYPE_H
#define ESBMC_JIMPLE_TYPE_H

#include <jimple-frontend/AST/jimple_ast.h>

// TODO: Specialize this class
class jimple_type : public jimple_ast {
public:
  virtual void from_json(const json& j) override;
  virtual std::string to_string() const override;
protected:
  std::string name; // e.g. int[][][][][] => name = int
  short dimensions; // e.g. int[][][][][] => dimensions = 5

};

#endif //ESBMC_JIMPLE_TYPE_H
