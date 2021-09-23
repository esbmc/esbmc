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
  virtual exprt to_exprt(const messaget &msg, contextt &ctx) const
  {
    msg.debug("Converting Jimple AST");
    exprt x;
    return x;
  };
  /**
   * Converts the object into a string
   * @return a human readable string of the object
   */
  virtual std::string to_string() const = 0;
};

void from_json(const json& j, jimple_ast& p);
void to_json(json&, const jimple_ast&);
#endif //ESBMC_JIMPLE_AST_H
