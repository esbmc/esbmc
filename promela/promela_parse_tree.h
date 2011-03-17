/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_PROMELA_PARSE_TREE_H
#define CPROVER_PROMELA_PARSE_TREE_H

#include <hash_cont.h>
#include <string_hash.h>
#include <expr.h>

class promela_parse_treet
{
 public:
  // the (global) declrataions   
  typedef std::list<exprt> declarationst;
  declarationst declarations;

  void swap(promela_parse_treet &promela_parse_tree);
  void clear();
};

#endif
