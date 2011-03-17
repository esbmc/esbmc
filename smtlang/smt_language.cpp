/*******************************************************************\

Module: SMT-LIB Frontend

Author: CM Wintersteiger

\*******************************************************************/

#include <message_stream.h>

#include "smt_language.h"
#include "smt_parser.h"
#include "smt_typecheck.h"
#include "smt_typecheck_expr.h"
#include "smt_link.h"
#include "smt_logics.h"
#include "expr2smt.h"

/*******************************************************************\

Function: smt_languaget::parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smt_languaget::parse(
  std::istream &instream,
  const std::string &path,
  message_handlert &message_handler)
{
  smt_parser.clear();

  smt_parser.filename=path;
  smt_parser.in=&instream;
  smt_parser.set_message_handler(&message_handler);

  bool result=smt_parser.parse();

  if(result==0) {
    parse_tree.swap(smt_parser.parse_tree);
    
//    if (parse_tree.theories.size()>0) {
//      smt_parse_treet::theoryt &t = parse_tree.theories.front();
//      std::cout << "Theory loaded: " << t.name << std::endl;
//    }
//    
//    if (parse_tree.logics.size()>0) {
//      smt_parse_treet::logict &l = parse_tree.logics.front();
//      std::cout << "Logic loaded: " << l.name << std::endl;
//    }
    
//    if (parse_tree.benchmarks.size()>0) {
//      smt_parse_treet::benchmarkt &b = parse_tree.benchmarks.front();
//      std::cout << "| SMT Benchmark Data" << std::endl;
//      std::cout << "|-------------------" << std::endl;
//      std::cout << "| Name: " << b.name << std::endl;
//      std::cout << "| Status: ";
//      if (b.status.front()==smt_parse_treet::benchmarkt::SAT) std::cout << "sat" << std::endl;
//      else if (b.status.front()==smt_parse_treet::benchmarkt::UNSAT) std::cout << "unsat" << std::endl;
//      else std::cout << "unknown" << std::endl;
//      std::cout << "| Logic: " << b.logics.front() << std::endl;
//      std::cout << "| # of notes: " << b.notes.size() << std::endl;
//      std::cout << "| # of extrasorts: " << b.sort_symbols.size() << std::endl;
//      std::cout << "| # of extrapreds: " << b.predicate_symbols.size() << std::endl;
//      std::cout << "| # of extrafuns: " << b.function_symbols.size() << std::endl;
//      std::cout << "| # of annotations: " << b.annotations.size() << std::endl;
//      std::cout << "|-------------------" << std::endl;
//    }
  }

  // save some memory
  smt_parser.clear();
  
  return result;
}
             
/*******************************************************************\

Function: smt_languaget::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smt_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  contextt new_context;
  
  smt_typecheckt checker(context, module, message_handler, parse_tree);
  if(checker.typecheck_main())
    return true;
  
  if(smt_link(context, new_context , message_handler, module))
    return true;
    
  return false;
}

/*******************************************************************\

Function: smt_languaget::final

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smt_languaget::final(
  contextt &context,
  message_handlert &message_handler)
{
  // now that we should have theories and everything,
  // typecheck all the formulas
  symbolst::iterator si = context.symbols.find(smt_typecheckt::bsymn);

  if (si!=context.symbols.end())
  {
    symbolt &s = si->second;

    forall_operands(it, s.value)
    {
      const exprt &benchmark = *it;
      std::string l=ii2string(benchmark.find("logic"));
      
      smt_finalizert* flzr = create_smt_finalizer(l, context, message_handler);
//      message_handler.print(8, "Using finalizer '" + flzr->logic() + 
//                               "' for benchmark '" +
//                               ii2string(benchmark.find("name")) + "'.");
      bool res = flzr->finalize(benchmark);
      delete flzr;
      if (res) return true;
    }
  }
  
  return false;
}

/*******************************************************************\

Function: smt_languaget::show_parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
  
void smt_languaget::show_parse(std::ostream &out)
{

}

/*******************************************************************\

Function: smt_languaget::from_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smt_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  return expr2smt(expr, code);
}

/*******************************************************************\

Function: smt_languaget::from_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smt_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  return type2smt(type, code);
}

/*******************************************************************\

Function: smt_languaget::to_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smt_languaget::to_expr(
  const std::string &code,
  const std::string &module,
  exprt &expr,
  message_handlert &message_handler,
  const namespacet &ns)
{
  messaget message(message_handler);
  message.error("to_expr not yet implemented");
  return true;
}

/*******************************************************************\

Function: new_smt_language

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
  
languaget *new_smt_language()
{
  return new smt_languaget;
}
