/*******************************************************************\

Module: SMT-LIB Frontend, Typechecking of expressions

Author: CM Wintersteiger

\*******************************************************************/

#ifndef SMT_TYPECHECK_EXPR_H_
#define SMT_TYPECHECK_EXPR_H_

#include <context.h>
#include <expr.h>
#include <message_stream.h>
#include <vector>
#include <hash_cont.h>

#include "smt_strings.h"

class smt_typecheck_exprt : protected smt_stringst
{
protected:
  const symbolst &theories;
  const exprt &benchmark;
  message_streamt &message_stream;
  
  typedef hash_map_cont<irep_idt, std::list<typet>, irep_id_hash>
    benchmark_functionst;  
  typedef hash_map_cont<irep_idt, std::list<typet>, irep_id_hash>
    benchmark_predicatest;
    
  typedef hash_map_cont<irep_idt, std::list<typet>, irep_id_hash>
    theory_functionst;
  typedef hash_map_cont<irep_idt, std::list<typet>, irep_id_hash>
    theory_predicatest;
  typedef hash_map_cont<irep_idt, 
            std::pair<theory_functionst,theory_predicatest>, irep_id_hash>
    theories_mapst;
  
    
  benchmark_functionst benchmark_functions;  
  benchmark_predicatest benchmark_predicates;  
  theories_mapst theories_maps;
  
  class scopet
  {
  public:
    typedef hash_map_cont<irep_idt, exprt, irep_id_hash> idst;
    idst ids;

    typedef hash_map_cont<irep_idt, exprt, irep_id_hash> fdefst;
    fdefst fdefs;
  };
  
  void build_hashtables(void);
  
  typedef std::list<scopet> scopest;
  scopest scopes;
  scopet& current_scope() { return scopes.back(); }
  void new_scope() { scopes.push_back(scopet()); }
  void pop_scope() { scopes.pop_back(); }
  
  typet search_var(const exprt&);
  virtual typet search_sort(const typet&);
  virtual typet search_fun(const exprt&, const typet&);
  virtual void check_result(const exprt&, const typet&, typet&);
  virtual void check_pre(const exprt&);
  
public:
  smt_typecheck_exprt(message_streamt &eh, const exprt& b, const symbolst &t);
    
  virtual ~smt_typecheck_exprt( void ) {};
  
  virtual typet typecheck_expr(const exprt&);
};

#endif /*SMT_TYPECHECK_EXPR_H_*/
