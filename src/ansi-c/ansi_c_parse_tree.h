/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ANSI_C_PARSE_TREE_H
#define CPROVER_ANSI_C_PARSE_TREE_H

#include <ansi-c/ansi_c_declaration.h>

class ansi_c_parse_treet
{
public:
  // the declarations
  typedef std::list<ansi_c_declarationt> declarationst;
  declarationst declarations;

  void swap(ansi_c_parse_treet &other);
  void clear();

  void output(std::ostream &out) const
  {
    for(const auto &declaration : declarations)
    {
      symbolt tmp;
      declaration.to_symbol(tmp);
      out << tmp;
    }
  }
};

#endif
