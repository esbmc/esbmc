//
// Created by rafaelsamenezes on 21/09/2021.
//

#ifndef ESBMC_JIMPLE_DECLARATION_H
#define ESBMC_JIMPLE_DECLARATION_H

#include <jimple-frontend/AST/jimple_method_body.h>
#include <jimple-frontend/AST/jimple_type.h>

class jimple_declaration : public jimple_method_field
{
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

protected:
  jimple_type t;
  std::string name;
};

#endif //ESBMC_JIMPLE_DECLARATION_H
