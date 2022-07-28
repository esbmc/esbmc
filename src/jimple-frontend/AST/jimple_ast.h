#ifndef ESBMC_JIMPLE_AST_H
#define ESBMC_JIMPLE_AST_H

#include <util/expr.h>
#include <util/context.h>
#include <util/std_types.h>
#include <util/c_types.h>
#include <nlohmann/json.hpp>

// For json parsing
using json = nlohmann::json;

/**
 * @brief Base interface for Jimple AST
 */
class jimple_ast
{
public:
  /**
   * @brief Prints the contents into the
   * the stdout.
   */
  void dump() const
  {
    log_debug("{}", this->to_string());
  }

  /**
   * @brief Initializes the current instance by parsing
   * a JSON
   *
   * @param json The json object relative to the structure
   */
  virtual void from_json(const json &) = 0;

  /**
   * @brief Converts the object into a string
   *
   * @return a human readable string of the object
   */
  virtual std::string to_string() const = 0;

protected:
  /**
   * @brief creates a symbol with the default characteristics
   *
   * TODO: Right now, we are setting it as C mode. In time, we
   * should convert it to Jimple.
   * By using C mode, we don't have to worry about typecast and
   * primitives for now.
   *
   * We still need to create every intrinsic type and variable for JVM
   *
   * @return an initialized symbolt
   */
  static symbolt create_jimple_symbolt(
    const typet &t,
    const std::string &module,
    const std::string &name,
    const std::string &id,
    const std::string function_name = "")
  {
    symbolt symbol;
    symbol.mode = "C";
    symbol.module = module;
    symbol.location = std::move(get_location(module, function_name));
    symbol.type = std::move(t);
    symbol.name = name;
    symbol.id = id;
    return symbol;
  }

  /**
   * @brief Create a temporary variable to be used
   *
   * @param t type of the variable
   * @param class_name class package to the variable
   * @param function_name function where the variable is going to be created
   * @return symbolt
   */
  static symbolt get_temp_symbol(
    const typet &t,
    const std::string &class_name,
    const std::string &function_name)
  {
    static unsigned int counter = 0;

    std::string id, name;
    id = get_symbol_name(
      class_name,
      function_name,
      "return_value$tmp$" + std::to_string(counter++));
    name = "return_value$tmp$";
    name += counter;
    auto tmp_symbol =
      create_jimple_symbolt(t, class_name, name, id, function_name);

    tmp_symbol.lvalue = true;
    tmp_symbol.static_lifetime = false;
    tmp_symbol.is_extern = false;
    tmp_symbol.file_local = true;
    return tmp_symbol;
  }

  /**
   * @brief Get the allocation function symbol
   *
   * This is going to be the function to be called
   * for `new` and `newarray` calls
   *
   * @return symbolt
   */
  static symbolt get_allocation_function()
  {
    std::string allocation_function = "malloc";
    code_typet code_type;
    code_type.return_type() = pointer_typet(empty_typet());
    code_type.arguments().push_back(uint_type());
    symbolt symbol;
    symbol.mode = "C";
    symbol.type = code_type;
    symbol.name = allocation_function;
    symbol.id = allocation_function;
    symbol.is_extern = false;
    symbol.file_local = false;
    return symbol;
  }

  /**
   * @brief Get the esbmc_get_object_size symbol
   *
   * This is going to be the function to be called
   * for `lengthof`
   *
   * @return symbolt
   */
  static symbolt get_lengthof_function()
  {
    std::string func = "__ESBMC_get_object_size";
    code_typet code_type;
    code_type.return_type() = uint_type();
    code_type.arguments().push_back(pointer_typet(empty_typet()));
    symbolt symbol;
    symbol.mode = "C";
    symbol.type = code_type;
    symbol.name = func;
    symbol.id = func;
    symbol.is_extern = false;
    symbol.file_local = false;
    return symbol;
  }

  /**
   * @brief Get the unique method name
   *
   * This is the full id for the method,
   *
   * In jimple this will mean `class_name:function_name`
   *
   * @param class_name
   * @param function_name
   * @return std::string
   */
  static std::string
  get_method_name(std::string class_name, std::string function_name)
  {
    std::ostringstream oss;
    oss << class_name << ":" << function_name;
    return oss.str();
  }

  /**
   * @brief Get the symbol name id
   *
   * Note: Jimple is already in SSA form
   *
   * @param class_name
   * @param function_name
   * @param symbol
   * @return std::string
   */
  static std::string get_symbol_name(
    std::string class_name,
    std::string function_name,
    std::string symbol)
  {
    std::ostringstream oss;
    oss << get_method_name(class_name, function_name) << "@" << symbol;
    return oss.str();
  }

  /**
   * @brief initialize a location for the symbol
   *
   * @return the location of a symbol (file, class, function)
   */
  static locationt
  get_location(const std::string &module, const std::string function_name = "")
  {
    locationt l;
    l.set_file(module + ".jimple");
    if(!function_name.empty())
      l.set_function(function_name);
    return l;
  }

public:
  /// instruction location
  int line_location = -1;
};

// These functions are used by nlohmann::json. Making it easier to
// parse the JSON file. Avoid using them directly.
void from_json(const json &j, jimple_ast &p);
void to_json(json &, const jimple_ast &);
#endif //ESBMC_JIMPLE_AST_H
