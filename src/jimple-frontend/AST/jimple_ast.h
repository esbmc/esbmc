/*******************************************************************\
Module: Jimple AST Interface
Author: Rafael Sá Menezes
Date: September 2021
Description: This interface will define every method that needs to
  be implemented by every Jimple AST
\*******************************************************************/

#ifndef ESBMC_JIMPLE_AST_H
#define ESBMC_JIMPLE_AST_H

#include <util/message/default_message.h>
#include <util/expr.h>
#include <util/context.h>
#include <nlohmann/json.hpp>

// For json parsing
using json = nlohmann::json;

/**
 * @brief Base interface for Jimple AST
 */
class jimple_ast {
public:
  /**
   * @brief Prints the contents into the
   * the stdout.
   */
  void dump() const
  {
    default_message msg;
    msg.debug(this->to_string());
  }

  /**
   * @brief Initializes the current instance by parsing
   * a JSON
   *
   * @param json The json object relative to the structure
   */  
  virtual void from_json(const json&) = 0;

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
   * @return an initialized symbolt
   */
  static symbolt create_jimple_symbolt(const typet &t, const std::string &module, const std::string &name, const std::string &id, const std::string function_name = "") {
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
   * @brief initialize a location for the symbol
   *
   * @return the location of a symbol (file, class, function)
   */
  static locationt get_location(const std::string &module, const std::string function_name = "")
  {
    locationt l;
    l.set_file(module+ ".jimple");
    if(!function_name.empty())
      l.set_function(function_name);
    return l;
  }
};

// These functions are used by nlohmann::json. Making it easier to
// parse the JSON file. You shouldn't use them directly.
void from_json(const json& j, jimple_ast& p);
void to_json(json&, const jimple_ast&);
#endif //ESBMC_JIMPLE_AST_H
