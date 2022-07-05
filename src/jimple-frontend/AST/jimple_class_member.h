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

  const std::string &getName() const
  {
    return name;
  }

  const std::string &getThrows() const
  {
    return throws;
  }
  const jimple_modifiers &getM() const
  {
    return m;
  }

  const jimple_type &getT() const
  {
    return t;
  }

  const std::shared_ptr<jimple_method_body> &getBody() const
  {
    return body;
  }

  const std::vector<std::shared_ptr<jimple_type>> &getParameters() const
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
  jimple_modifiers m;
  std::string throws;
  jimple_type t;
  std::shared_ptr<jimple_method_body> body;
  std::vector<std::shared_ptr<jimple_type>> parameters;
};

// TODO: Class Field
#endif //ESBMC_JIMPLE_CLASS_MEMBER_H
