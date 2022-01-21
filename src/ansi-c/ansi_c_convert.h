/*******************************************************************\

Module: ANSI-C Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ANSI_C_CONVERT_H
#define CPROVER_ANSI_C_CONVERT_H

#include <ansi-c/ansi_c_parse_tree.h>
#include <ansi-c/c_storage_spec.h>
#include <util/message/message_stream.h>
#include <util/std_code.h>
#include <iostream>

bool ansi_c_convert(
  ansi_c_parse_treet &ansi_c_parse_tree,
  const std::string &module,
  const messaget &message_handler);

bool ansi_c_convert(
  exprt &expr,
  const std::string &module,
  const messaget &message_handler);

class ansi_c_convertt : public message_streamt
{
public:
  ansi_c_convertt(const std::string &_module, const messaget &_message_handler)
    : message_streamt(_message_handler), module(_module)
  {
  }

  virtual void convert(ansi_c_parse_treet &ansi_c_parse_tree);

  // expressions
  virtual void convert_expr(exprt &expr);

  // types
  virtual void convert_type(typet &type);

  virtual void convert_type(typet &type, c_storage_spect &c_storage_spec);

protected:
  virtual void convert_code(codet &code);

  virtual void convert_declaration(ansi_c_declarationt &declaration);

  const std::string &module;

  irep_idt final_id(const irep_idt &id) const
  {
    return id2string(id);
  }
};

#endif
