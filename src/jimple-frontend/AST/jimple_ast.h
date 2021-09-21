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
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class jimple_ast {
public:
  /**
   * Prints the contents into the
   * the stdout.
   */
  void dump() {
    default_message msg;
    msg.debug(this->to_string());
  }

  virtual void from_json(const json&) = 0;

  /**
   * Converts the object into a string
   * @return a human readable string of the object
   */
  virtual std::string to_string() = 0;
};

void from_json(const json& j, jimple_ast& p);
void to_json(json&, const jimple_ast&);
#endif //ESBMC_JIMPLE_AST_H
