/*******************************************************************\

Module: ANSI-C Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ANSI_C_TYPECHECK_H
#define CPROVER_ANSI_C_TYPECHECK_H

#include <ansi-c/ansi_c_parse_tree.h>
#include <ansi-c/c_typecheck_base.h>

bool ansi_c_typecheck(
  ansi_c_parse_treet &parse_tree,
  contextt &context,
  const std::string &module);

class ansi_c_typecheckt : public c_typecheck_baset
{
public:
  ansi_c_typecheckt(
    ansi_c_parse_treet &_parse_tree,
    contextt &_context,
    const std::string &_module)
    : c_typecheck_baset(_context, _module), parse_tree(_parse_tree)
  {
  }

  ~ansi_c_typecheckt() override = default;

  void typecheck() override;

protected:
  ansi_c_parse_treet &parse_tree;
};

#endif
