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

  const std::string &getTName() const
  {
    return name;
  }

  const std::string &getTMode() const
  {
    return mode;
  }

  const short &getTDim() const
  {
    return dimensions;
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

  private:  
  enum class BASE_TYPES {
    INT,
    VOID,
    OTHER
  };
  BASE_TYPES bt;
  std::map<std::string, BASE_TYPES> from_map = {
    {"int", BASE_TYPES::INT},
    {"void", BASE_TYPES::VOID},
    {"__other", BASE_TYPES::OTHER}};
};

#endif //ESBMC_JIMPLE_TYPE_H
