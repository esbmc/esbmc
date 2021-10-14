//
// Created by rafaelsamenezes on 21/09/2021.
//

#ifndef ESBMC_JIMPLE_TYPE_H
#define ESBMC_JIMPLE_TYPE_H

#include <jimple-frontend/AST/jimple_ast.h>
#include <util/std_code.h>
#include <util/c_types.h>
#include <util/expr_util.h>

// TODO: Specialize this class
class jimple_type : public jimple_ast
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override;
  virtual typet to_typet() const;

  bool is_array() const
  {
    return dimensions > 0;
  }

protected:
  std::string name; // e.g. int[][][][][] => name = int
  std::string mode;
  short dimensions; // e.g. int[][][][][] => dimensions = 5

  typet get_base_type() const;
  typet get_builtin_type() const;

  // TODO: Support for matrix
  typet get_arr_type() const
  {
    typet base = get_base_type();
    return array_typet(base, gen_one(index_type()));
  }
};

#endif //ESBMC_JIMPLE_TYPE_H
