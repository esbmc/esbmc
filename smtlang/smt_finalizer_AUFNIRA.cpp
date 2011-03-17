/*******************************************************************\

Module: SMT-LIB Builtin Logics, Finalizer for the AUFNIRA logic  

Author: CM Wintersteiger

\*******************************************************************/

#include "smt_typecheck.h"
#include "smt_typecheck_expr.h"
#include "expr2smt.h"

#include "builtin_theories.h"
#include "smt_finalizer_AUFNIRA.h"

/*******************************************************************\

Function: smt_finalizer_AUFNIRAt::finalize

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smt_finalizer_AUFNIRAt::finalize(const exprt& benchmark)
{
  message_streamt err(message_handler);
      
  std::string l=ii2string(benchmark.find("logic"));  
    
  if (l!=logic())
  {
    err.str << "The " << logic() << " finalizer cannot finalize " <<
               "benchmark `" << ii2string(benchmark.find("name")) << "', " <<
               "because its logic is a different one ('" << l << "')";
    err.error();
    return true;
  }

  std::string theory="Int_Int_Real_Array_ArraysEx";
  
  symbolst empty;
  check_double_sorts(benchmark, empty, err);
  check_double_functions(benchmark, empty, err);
  check_double_predicates(benchmark, empty, err);
  
  smt_typecheck_expr_AUFNIRAt checker(err, benchmark, empty);

  const exprt &assumptions =
    static_cast<const exprt&>(benchmark.find("assumptions"));

  forall_operands(ait, assumptions)
    checker.typecheck_expr(*ait);

  const exprt &formulas =
    static_cast<const exprt&>(benchmark.find("formulas"));

  forall_operands(fit, formulas)
    checker.typecheck_expr(*fit);
        
  return err.get_error_found();  
}

/*******************************************************************\

Function: smt_finalizer_AUFLIAt::check_double_sorts

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_finalizer_AUFNIRAt::check_double_sorts(
  const exprt& benchmark, 
  const symbolst &theories,
  message_streamt &err) 
{
  const typet &sorts=
    static_cast<const typet &>(benchmark.find("sorts"));

  forall_subtypes(sit, sorts)
  {
    if(smt_theory_Int_Int_Real_Array_ArraysEx().check_sorts(ii2string(*sit)))
    {
      err.str <<
        "Sort symbol `" << ii2string(*sit) << "' defined twice"
        " in benchmark `" << ii2string(benchmark.find("name")) << "'";
      err.error();
    }
  }
}

/*******************************************************************\

Function: smt_finalizer_AUFNIRAt::check_double_functions

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_finalizer_AUFNIRAt::check_double_functions(
  const exprt &benchmark, 
  const symbolst &theories,
  message_streamt &message) 
{
  // as of Definition 5, item 5 of the SMT-LIB Standard v1.2
  const typet &funs=static_cast<const typet &>
    (benchmark.find("functions"));
    
  forall_subtypes(fit, funs)
  { 
    std::string fname = ii2string(*fit);
    typet params;
    params.subtypes() = fit->subtypes(); // copy!
    
    typet retval = params.subtypes().back();
    params.subtypes().pop_back(); // remove last
    
    typet t = 
      smt_theory_Int_Int_Real_Array_ArraysEx().check_functions( 
        ii2string(*fit), params);
    
    if (t.id()!=(""))
    {
      if(retval!=t)
      {
        message.str <<
          "Function symbol `" << fname << "' defined twice"
          " with different return types in benchmark `" <<
          ii2string(benchmark.find("name")) << "'";
        message.error();
      }
      else // they're equal
      {
        message.str <<
          "Function symbol `" << fname << "' defined twice"
          " in benchmark `" << ii2string(benchmark.find("name")) << "'";
        message.error();
      }    
    }
  }
}

/*******************************************************************\

Function: smt_finalizer_AUFNIRAt::check_double_predicates

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_finalizer_AUFNIRAt::check_double_predicates(
  const exprt& benchmark, 
  const symbolst& theories,
  message_streamt &err) 
{
  const typet &preds=static_cast<const typet&>
    (benchmark.find("predicates"));

  forall_subtypes(pit, preds)
  { // for all functions
    typet params;
    params.subtypes() = pit->subtypes();
    if(smt_theory_Int_Int_Real_Array_ArraysEx().check_predicates( 
        ii2string(*pit), params))
    {
      err.str <<
        "predicate symbol `" << ii2string(*pit) << "' defined twice"
        " in benchmark `" << ii2string(benchmark.find("name")) << "'";
      err.warning();
    }
  }
}

/*******************************************************************\

Function: smt_typecheck_expr_AUFNIRAt::search_fun

  Inputs: an exprt and a list of typets

 Outputs: a typet

 Purpose: searches the benchmark and the theory for a fitting function
          (name in e.id() and signature in the params list)

\*******************************************************************/

typet smt_typecheck_expr_AUFNIRAt::search_fun(
  const exprt &e,
  const typet &params) 
{  
  if (smt_theory_Int_Int_Real_Array_ArraysEx().check_predicates( 
                            ii2string(e), params))
    return typet("bool");
  else   
  {
    typet t = 
      smt_theory_Int_Int_Real_Array_ArraysEx().check_functions( 
                            ii2string(e), params);
    if (t.id()!="") return t; 
  }
  
  return smt_typecheck_exprt::search_fun(e, params);
}

/*******************************************************************\

Function: smt_typecheck_expr_AUFNIRAt::search_sort

  Inputs: an exprt 

 Outputs: nothing

 Purpose: 

\*******************************************************************/

typet smt_typecheck_expr_AUFNIRAt::search_sort(const typet& sort)
{  
  if (smt_theory_Int_Int_Real_Array_ArraysEx().check_sorts(ii2string(sort)))
    return typet(sort.id_string());
  return smt_typecheck_expr_generict::search_sort(sort);
}
