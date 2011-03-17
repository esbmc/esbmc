/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_PROMELA_PARSER_H
#define CPROVER_PROMELA_PARSER_H

#include <parser.h>

#include "promela_parse_tree.h"

typedef enum { PR_UNKNOWN, PR_SYMBOL, PR_TYPEDEF, PR_BEHAVIOR,
               PR_CHANNEL, PR_INTERFACE } promela_id_classt;

int yypromelaparse();

class promela_parsert:public parsert
{
public:
  promela_parse_treet parse_tree;
  
  virtual bool parse()
  {
    return yypromelaparse();
  }

  enum { LANGUAGE, EXPRESSION } grammar;

  #if 0
  virtual void clear()
  {
    parsert::clear();
    parse_tree.clear();
    
    string_literal.clear();
    
    // setup global scope
    scopes.clear();
    // this is the global scope
    scopes.push_back(scopet());
  }

 public:
  // internal state
  std::string string_literal;
  
  class identifiert
  {
   public:
    promela_id_classt id_class;
  };
 
  class scopet
  {
   public:
    typedef std::map<std::string, identifiert> name_mapt;
    name_mapt name_map;
    
    std::string prefix;
    
    unsigned compound_counter;
    
    scopet():compound_counter(0) { }
    
    void swap(scopet &scope)
    {
      name_map.swap(scope.name_map);
      prefix.swap(scope.prefix);
      std::swap(compound_counter, scope.compound_counter);
    }

    promela_parse_treet::declarationst declarations;
  };
   
  typedef std::list<scopet> scopest;
  scopest scopes;
  
  // save for function definitions
  scopet last_parameter_list_scope;

  void pop_scope()
  {
    scopes.pop_back();
  }
   
  scopet &current_scope()
  {
    assert(!scopes.empty());
    return scopes.back();
  }
   
  void move_declaration(irept &declaration)
  {
    if(current_scope().prefix.empty()) // global?
    {
      parse_tree.declarations.push_back();
      parse_tree.declarations.back().swap(declaration);
    }
    else
    {
      current_scope().declarations.push_back();
      current_scope().declarations.back().swap(declaration);    
    }
  }
   
  void new_scope(const std::string &prefix)
  {
    const scopet &current=current_scope();
    scopes.push_back(scopet());
    scopes.back().prefix=current.prefix+prefix;
  }

  promela_id_classt lookup(std::string &name) const;
  #endif
};

extern promela_parsert promela_parser;

int yypromelaerror(const std::string &error);
void promela_scanner_init();

#endif
