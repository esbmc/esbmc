/*******************************************************************\
Module: Jimple File AST
Author: Rafael Sá Menezes
Date: September 2021
Description: Jimple File AST parser and holder
\*******************************************************************/

#ifndef ESBMC_JIMPLE_FILE_H
#define ESBMC_JIMPLE_FILE_H

#include <jimple-frontend/AST/jimple_ast.h>
#include <jimple-frontend/AST/jimple_modifiers.h>
#include <jimple-frontend/AST/jimple_class_member.h>
#include <util/expr.h>

/**
 * @brief Main AST for Class/Interface
 * 
 * Every Jimple file contains at most one Class/Interface, this class
 * will represent this.
 *
 * The file can:
 * - extend (or implement) another Class/Interface
 * - contain modifiers: public/private/static/etc... 
 *   (see jimple_modifiers.hpp)
 * - contain methods and attributes.
 */
class jimple_file : public jimple_ast
{
public:
  // overrides
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override;

  /**
   * @brief initializes the object with a file
   *
   * @param path A path to a .jimple file
   */
  void load_file(const std::string &path);

  // A file can be a class or interface
  enum class file_type
  {
    Class,
    Interface
  };

  // Get the file_type using a string
  file_type from_string(const std::string &name) const;

  // Convert file_type into a string
  std::string to_string(const file_type &ft) const;

  /**
   * @brief convert the object into a exprt
   */
  virtual exprt to_exprt(contextt &ctx) const;

  file_type get_mode() const
  {
    return mode;
  }
  const std::string &get_class_name() const
  {
    return class_name;
  }
  const std::string &get_extends() const
  {
    return extends;
  }
  const std::string &get_implements() const
  {
    return implements;
  }
  const jimple_modifiers &get_modifiers() const
  {
    return modifiers;
  }
  const std::vector<jimple_class_method> &get_body() const
  {
    return body;
  }

  bool is_interface() const
  {
    return get_mode() == file_type::Interface;
  }

protected:
  file_type mode;
  std::string class_name;
  std::string extends;
  std::string implements;
  jimple_modifiers modifiers;
  std::vector<jimple_class_method> body;

private:
  std::map<std::string, file_type> from_map = {
    {"Class", file_type::Class},
    {"Interface", file_type::Interface}};

  std::map<file_type, std::string> to_map = {
    {file_type::Class, "Class"},
    {file_type::Interface, "Interface"}};
};
#endif //ESBMC_JIMPLE_FILE_H
