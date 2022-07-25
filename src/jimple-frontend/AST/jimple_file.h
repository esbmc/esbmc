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
 *
 * Notes:
 * - Java/Kotlin have support for inner classes. When compiled into bytecode
 *   each class has its own .class file. For example, an inner class Bar
 *   inside the Foo class will generate both Foo.class and Foo$Bar.class.
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

  bool is_interface() const
  {
    return mode == file_type::Interface;
  }
  file_type mode;
  std::string class_name;
  std::string extends;
  std::string implements;
  jimple_modifiers modifiers;
  std::vector<std::shared_ptr<jimple_class_member>> body;

private:
  std::map<std::string, file_type> from_map = {
    {"Class", file_type::Class},
    {"Interface", file_type::Interface}};

  std::map<file_type, std::string> to_map = {
    {file_type::Class, "Class"},
    {file_type::Interface, "Interface"}};
};
#endif //ESBMC_JIMPLE_FILE_H
