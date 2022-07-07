/*******************************************************************\

Module: ANSI-C Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/ansi_c_typecheck.h>

void ansi_c_typecheckt::typecheck()
{
  for(auto &declaration : parse_tree.declarations)
  {
    symbolt symbol;
    declaration.to_symbol(symbol);
    typecheck_symbol(symbol);
  }
}

bool ansi_c_typecheck(
  ansi_c_parse_treet &ansi_c_parse_tree,
  contextt &context,
  const std::string &module)
{
  ansi_c_typecheckt ansi_c_typecheck(ansi_c_parse_tree, context, module);
  return ansi_c_typecheck.typecheck_main();
}
