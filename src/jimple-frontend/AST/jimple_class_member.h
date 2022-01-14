/*******************************************************************\
Module: Jimple Class Member Interface
Author: Rafael Sá Menezes
Date: September 2021
Description: This interface will hold anything 
  that belongs to a Jimple class
\*******************************************************************/

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
};

// TODO: class_field
// TODO: class_definition

/**
 * @brief A class (or interface) method of a Jimple file
 * 
 * example:
 * 
 * class Foo {
 *   public void jimple_class_method {
 *      do_stuff();
 *   }
 * }
 * 
 */
class jimple_class_method : public jimple_class_member
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &file_name) const;

  const std::string &get_name() const
  {
    return name;
  }

  const std::string &get_throws() const
  {
    return throws;
  }
  const jimple_modifiers &get_modifiers() const
  {
    return modifiers;
  }

  const jimple_type &get_type() const
  {
    return type;
  }

  const std::shared_ptr<jimple_method_body> &get_body() const
  {
    return body;
  }

  const std::vector<std::shared_ptr<jimple_type>> &get_parameters() const
  {
    return parameters;
  }

protected:
  // We need an unique name for each function
  std::string get_hash_name() const
  {
    // TODO: use some hashing to also use the types
    // TODO: DRY
    return std::to_string(parameters.size());
  }
  std::string name;
  jimple_modifiers modifiers;
  std::string throws;
  jimple_type type;
  std::shared_ptr<jimple_method_body> body;
  std::vector<std::shared_ptr<jimple_type>> parameters;
};

// TODO: Class Field
#endif //ESBMC_JIMPLE_CLASS_MEMBER_H
