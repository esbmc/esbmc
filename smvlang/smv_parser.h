/*******************************************************************\

Module: SMV Parser

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_SMV_PARSER_H
#define CPROVER_SMV_PARSER_H

#include <parser.h>

#include "smv_parse_tree.h"

int yysmvparse();

class smv_parsert:public parsert
{
 public:
  smv_parse_treet parse_tree;
  smv_parse_treet::modulet *module;
  
  virtual bool parse()
  {
    return yysmvparse();
  }
  
  virtual void clear()
  {
    parsert::clear();
    parse_tree.clear();
    module=NULL;
  }
};

extern smv_parsert smv_parser;

#endif
