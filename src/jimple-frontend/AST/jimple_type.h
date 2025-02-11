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
  virtual typet to_typet(const contextt &ctx) const;

  bool is_array() const
  {
    return dimensions > 0;
  }

  std::string name; // e.g. int[][][][][] => name = int
  short dimensions; // e.g. int[][][][][] => dimensions = 5

protected:
  typet get_base_type(const contextt &ctx) const;
  typet get_builtin_type() const;

  typet get_arr_type(const contextt &ctx) const
  {
    typet base = get_base_type(ctx);
    typet ptr_type = pointer_typet(base);
    for (int i = 1; i < dimensions; i++)
      ptr_type = pointer_typet(ptr_type);

    return ptr_type;
  }

private:
  enum class BASE_TYPES
  {
    INT,
    BOOLEAN,
    _VOID,
    OTHER
  };
  BASE_TYPES bt;
  std::map<std::string, BASE_TYPES> from_map = {
    /* Basic JVM types  */
    {"int", BASE_TYPES::INT},
    {"byte", BASE_TYPES::INT},
    {"char", BASE_TYPES::INT},
    {"short", BASE_TYPES::INT},
    {"boolean", BASE_TYPES::INT},
    {"long", BASE_TYPES::INT},
    {"float", BASE_TYPES::INT},
    {"double", BASE_TYPES::INT},
    {"void", BASE_TYPES::_VOID},
    /* Basic Java classes that can work as primitive types */
    {"java.lang.Integer", BASE_TYPES::INT},
    {"java.util.Random",
     BASE_TYPES::
       INT}, // We dont really care about the initialization of this mode
    {"java.lang.String", BASE_TYPES::INT}, // TODO: handle this properly
    /* TODO: these are hacks and should be moved into an intrinsics class */
    {"Main", BASE_TYPES::INT},                     // TODO: handle this properly
    {"java.lang.AssertionError", BASE_TYPES::INT}, // TODO: handle this properly
    {"java.lang.Runtime", BASE_TYPES::INT},        // TODO: handle this properly
    {"java.lang.Class", BASE_TYPES::INT},          // TODO: handle this properly
    {"__other", BASE_TYPES::OTHER}};
};

#endif //ESBMC_JIMPLE_TYPE_H
