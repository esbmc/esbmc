//
// Created by rafaelsamenezes on 21/09/2021.
//

#ifndef ESBMC_JIMPLE_CLASS_MEMBER_H
#define ESBMC_JIMPLE_CLASS_MEMBER_H

#include <jimple-frontend/AST/jimple_ast.h>
#include <jimple-frontend/AST/jimple_modifiers.h>
#include <jimple-frontend/AST/jimple_type.h>
#include <jimple-frontend/AST/jimple_method_body.h>

/**
 * @brief This class will hold any member of a Jimple File
 *
 * This can be any attributes or methods that are inside the file
 */
class jimple_class_member : public jimple_ast
{
};

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

protected:
  std::string name;
  jimple_modifiers m;
  std::string throws;
  jimple_type t;
  std::shared_ptr<jimple_method_body> body;
  std::string parameters;
};

// TODO: Class Field
#endif //ESBMC_JIMPLE_CLASS_MEMBER_H
