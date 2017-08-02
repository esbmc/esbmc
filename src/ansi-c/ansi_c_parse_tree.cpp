/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/ansi_c_parse_tree.h>

void ansi_c_parse_treet::swap(ansi_c_parse_treet &ansi_c_parse_tree)
{
  ansi_c_parse_tree.declarations.swap(declarations);
}

void ansi_c_parse_treet::clear()
{
  declarations.clear();
}

