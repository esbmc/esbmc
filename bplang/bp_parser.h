/*******************************************************************\

Module: Boolean Program Parser

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BP_PARSER_H
#define CPROVER_BP_PARSER_H

#include <parser.h>

#include "bp_parse_tree.h"

int yybpparse();

class bp_parsert:public parsert
{
public:
  bp_parse_treet parse_tree;
  
  virtual bool parse()
  {
    return yybpparse();
  }
  
  virtual void clear()
  {
    parsert::clear();
    parse_tree.clear();
  }
  
  // for the locations
  irep_idt function;
};

extern bp_parsert bp_parser;

#endif
