/*******************************************************************\

Module: SMT-LIB Frontend, Typechecking

Author: CM Wintersteiger

\*******************************************************************/

#ifndef SMT_TYPECHECK_H_
#define SMT_TYPECHECK_H_

#include <context.h>
#include <message.h>
#include <typecheck.h>

#include "smt_parse_tree.h"

class smt_typecheckt: public typecheckt
{
private:
  contextt &context;
  const std::string &module;
  message_handlert &message_handler;
  smt_parse_treet &parse_tree;
  
public:
  static std::string prefix;
  static std::string tbase;
  static std::string lbase;
  static std::string bsymn;

  smt_typecheckt(
    contextt& c, 
    const std::string& m, 
    message_handlert& mh,
    smt_parse_treet& pt) :
    typecheckt(mh),
    context(c),
    module(m),
    message_handler(mh),
    parse_tree(pt)
  {
  }
    
  virtual void typecheck();
  
  static bool signature_eq( const typet& f1, const typet& f2 );
  static bool signature_eq_except_last( const typet& f1, const typet& f2 );
  
protected:
  void typecheck_theories();
  void typecheck_logics();
  void typecheck_benchmarks();
  
  void typecheck_theory_basics(const smt_parse_treet::theoryt&);
  void typecheck_theory_sorts(const smt_parse_treet::theoryt&, symbolt&);
  void typecheck_theory_functions(const smt_parse_treet::theoryt&, symbolt&);
  void typecheck_theory_predicates(const smt_parse_treet::theoryt&, symbolt&);
  void typecheck_theory_axioms(const smt_parse_treet::theoryt&, symbolt&);
  
  void typecheck_benchmark_basics(
    const smt_parse_treet::benchmarkt&);

  void typecheck_benchmark_logics(
    const smt_parse_treet::benchmarkt&, irept&);

  void typecheck_benchmark_sorts(
    const smt_parse_treet::benchmarkt&, irept&);

  void typecheck_benchmark_functions(
    const smt_parse_treet::benchmarkt&, irept&);

  void typecheck_benchmark_predicates(
    const smt_parse_treet::benchmarkt&, irept&);

  void typecheck_benchmark_assumptions(
    const smt_parse_treet::benchmarkt&, irept&);

  void typecheck_benchmark_formulas(
    const smt_parse_treet::benchmarkt&, irept&);
  
  smt_parse_treet::theoriest::const_iterator 
    find_theory(const std::string &);
  
  symbolt new_symbol(const std::string &name);
};
                    
#endif /*SMT_TYPECHECK_H_*/
