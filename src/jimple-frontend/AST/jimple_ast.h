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
using json = nlohmann::json;

class jimple_ast {
public:
  /**
   * Prints the contents into the
   * the stdout.
   */
  void dump() const
  {
    default_message msg;
    msg.debug(this->to_string());
  }

  virtual void from_json(const json&) = 0;

  /**
   * Converts the object into a string
   * @return a human readable string of the object
   */
  virtual std::string to_string() const = 0;

  protected:
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

  static locationt get_location(const std::string &module, const std::string function_name = "")
  {
    locationt l;
    //l.set_line(-1);
    l.set_file(module+ ".jimple");
    if(!function_name.empty())
      l.set_function(function_name);
    return l;
  }
};

void from_json(const json& j, jimple_ast& p);
void to_json(json&, const jimple_ast&);
#endif //ESBMC_JIMPLE_AST_H
