/*******************************************************************\

Module: Boolean Program Parse Tree

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BP_PARSE_TREE_H
#define CPROVER_BP_PARSE_TREE_H

#include <hash_cont.h>
#include <string_hash.h>
#include <expr.h>

class bp_parse_treet
{
 public:
  typedef std::list<exprt> declarationst;
  declarationst declarations;
  
  void swap(bp_parse_treet &bp_parse_tree);
  void clear();
};

#endif
