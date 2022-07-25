#ifndef ESBMC_JIMPLE_CLASS_MEMBER_H
#define ESBMC_JIMPLE_CLASS_MEMBER_H

#include <jimple-frontend/AST/jimple_ast.h>
#include <jimple-frontend/AST/jimple_modifiers.h>
#include <jimple-frontend/AST/jimple_type.h>
#include <jimple-frontend/AST/jimple_method_body.h>

#include <string>

/**
 * @brief This class will hold any member of a Jimple File
 *
 * This can be any attributes or methods that are inside the file
 */
class jimple_class_member : public jimple_ast
{
public:
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &file_name) const = 0;
};

/**
 * @brief A class (or interface) field of a Jimple file
 *
 * example:
 *
 * class Foo {
 *
 *   public int a;
 * }
 *
 */
class jimple_class_field : public jimple_class_member
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &file_name) const override;

  std::string name;
  jimple_modifiers modifiers;
  jimple_type type;
};

// TODO: class_definition

/**
 * @brief A class (or interface) method of a Jimple file
 *
 * example:
 *
 * class Foo {
 *   public void jimple_method {
 *      do_stuff();
 *   }
 * }
 *
 */
class jimple_method : public jimple_class_member
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &file_name) const override;

  std::string name;
  jimple_modifiers modifiers;
  std::string throws;
  jimple_type type;
  std::shared_ptr<jimple_method_body> body;
  std::vector<std::shared_ptr<jimple_type>> parameters;

protected:
  // We need an unique name for each function
  std::string get_hash_name() const
  {
    // TODO: use some hashing to also use the types
    int add = modifiers.is_static() ? 0 : 1;
    return std::to_string(parameters.size() + add);
  }
};
#endif //ESBMC_JIMPLE_CLASS_MEMBER_H
