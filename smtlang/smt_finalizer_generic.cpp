/*******************************************************************\

Module: SMT-LIB Builtin Logics, Generic Finalizer 

Author: CM Wintersteiger

\*******************************************************************/

#include "smt_typecheck.h"
#include "smt_typecheck_expr.h"
#include "expr2smt.h"

#include "smt_finalizer_generic.h"

/*******************************************************************\

Function: smt_finalizer_generict::finalize

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smt_finalizer_generict::finalize( const exprt& benchmark )
{
  message_streamt err(message_handler);  
  symbolst theories;
  std::string l=ii2string(benchmark.find("logic"));
  
  symbolst::iterator li=context.symbols.find(smt_typecheckt::lbase + l);
  if (li==context.symbols.end() || li->second.is_extern)
  {
    err.str << "undefined logic `" << l <<
               "' in benchmark `" << benchmark.get("name") << "'";
    err.error();
    return true;
  }

  typet &ts=static_cast<typet &>(li->second.type.add("theories"));

  forall_subtypes(tit, ts)
  {
    symbolst::iterator ti=
      context.symbols.find(smt_typecheckt::tbase + tit->id_string());

    if(ti==context.symbols.end() || ti->second.is_extern)
    {
      err.str <<
        "undefined theory: `" << tit->id() <<
        "' in benchmark `" << benchmark.get("name")
        << "'";
      err.error();
      return true;
    }

    theories.insert(*ti);
  }      
  
  check_double_sorts(benchmark, theories, err);
  check_double_functions(benchmark, theories, err);
  check_double_predicates(benchmark, theories, err);
  
  smt_typecheck_expr_generict checker(err, benchmark, theories);

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

Function: smt_finalizer_generict::check_double_sorts

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_finalizer_generict::check_double_sorts(
  const exprt& benchmark, 
  const symbolst &theories,
  message_streamt &err) 
{
  const typet &sorts=
    static_cast<const typet &>(benchmark.find("sorts"));

  forall_subtypes(sit, sorts)
    forall_symbols(tit, theories)
    {
      const typet &ts=static_cast<const typet &>
        (tit->second.type.find("sorts"));

      forall_subtypes(oit, ts)
      {
        if(oit->id()==sit->id())
        {
          err.str <<
            "Sort symbol `" << sit->id() << "' defined twice"
            " in benchmark `" << benchmark.get("name") << "'";
          err.error();
        }
      }
    }
}

/*******************************************************************\

Function: smt_finalizer_generict::check_double_functions

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_finalizer_generict::check_double_functions(
  const exprt &benchmark, 
  const symbolst &theories,
  message_streamt &message) 
{
  // as of Definition 5, item 5 of the SMT-LIB Standard v1.2
  const typet &funs=static_cast<const typet &>
    (benchmark.find("functions"));
  
  forall_subtypes(fit, funs)
  { // for all functions
    forall_symbols(tit, theories)
    { // check in all theories
      const typet &ts=static_cast<const typet &>
        (tit->second.type.find("functions"));
      
      forall_subtypes(oit, ts)
      { // if any of the functions
        if(oit->id()==fit->id())
        { // matches name ...
          check_double_function_signatures(benchmark, *oit, *fit, message);
        }
      }
    }
  }
}

/*******************************************************************\

Function: smt_finalizer_generict::check_double_function_signatures

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_finalizer_generict::check_double_function_signatures(
  const exprt& benchmark,
  const typet& f1,
  const typet& f2,
  message_streamt &err)
{
  bool equal=false; 
  bool equalexceptlast=
    smt_typecheckt::signature_eq_except_last(f1, f2);
  
  if(equalexceptlast && 
     f1.subtypes().back().id()==f2.subtypes().back().id())
  {
    equal=true;
  }
  
  if(equalexceptlast)
  {
    err.str <<
      "Function symbol `" << f1.id() << "' defined twice"
      " with different return types in benchmark `" <<
      benchmark.get("name") << "'";
    err.error();
  }
  else if(equal)
  {
    err.str <<
      "Function symbol `" << f1.id() << "' defined twice"
      " in benchmark `" << benchmark.get("name") << "'";
    err.error();
  }
}

/*******************************************************************\

Function: smt_finalizer_generict::check_double_predicates

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_finalizer_generict::check_double_predicates(
  const exprt& benchmark, 
  const symbolst& theories,
  message_streamt &err) 
{
  const typet &preds=static_cast<const typet&>
    (benchmark.find("predicates"));

  forall_subtypes(pit, preds)
  { // for all functions
    forall_symbols(tit, theories)
    { // check in all theories
      const typet &ts = static_cast<const typet &>
        (tit->second.type.find("predicates"));

      forall_subtypes(oit, ts)
      { // if any of the functions
        if (oit->id()==pit->id())
        { // matches name ...
          // only a warning, if the signatures match
          bool equal=true; 
          if (oit->subtypes().size()==pit->subtypes().size())
          {// ... and signature
            for (unsigned i=0; i<oit->subtypes().size(); i++)
            {
              if (oit->subtypes()[i].id()!=pit->subtypes()[i].id())
                equal=false;
            }
          }
          else
            equal=false;
          
          if(equal)
          {
            err.str <<
              "predicate symbol `" << pit->id() << "' defined twice"
              " in benchmark `" << benchmark.get("name") << "'";
            err.warning();
          }
        }
      }
    }
  }
}

/*******************************************************************\

Function: smt_typecheck_expr_generict::search_fun

  Inputs: an exprt and a list of typets

 Outputs: a typet

 Purpose: searches the benchmark and the theory for a fitting function
          (name in e.id() and signature in the params list)

\*******************************************************************/

typet smt_typecheck_expr_generict::search_fun(
  const exprt &e,
  const typet &params) 
{
  //std::cout << "Searching function: " << e.id();

  typet paramsplus = params;
  paramsplus.subtypes().push_back(typet());
  
  { // check theory-functions and predicates
    // (for now there is always just one theory, because
    // the standard doesn't allow subtheories.)
    
    for (theories_mapst::iterator tit = theories_maps.begin();
         tit != theories_maps.end();
         tit++)
    {
      
      theory_functionst::const_iterator tfit = tit->second.first.find(e.id_string());
      if (tfit!=tit->second.first.end()) 
      {
        const std::list<typet> &f = tfit->second;
        for (std::list<typet>::const_iterator it = f.begin();
             it!=f.end();
             it++ )
        {
          if (smt_typecheckt::signature_eq_except_last(*it, paramsplus))
          {
            return it->subtypes().back();
          }
        }
      }
      
      theory_predicatest::const_iterator tpit = tit->second.second.find(e.id_string());
      if (tpit!=tit->second.second.end()) 
      {
        const std::list<typet> &p = tpit->second;
        for (std::list<typet>::const_iterator it = p.begin();
             it!=p.end();
             it++ )
        {
          if (smt_typecheckt::signature_eq_except_last(*it, params))
          {
            return typet("bool");
          }
        }
      }
      
    }    
  }
  
  // theory-functions with annotations
  if (params.subtypes().size()>=2) 
  { 
    bool allthesame=true;
    for (unsigned i=1; i<params.subtypes().size(); i++)
    {
      if (params.subtypes()[i].id() !=
          params.subtypes()[i-1].id())
      {
        allthesame=false; break;
      }
    }
    
    if (allthesame)
      for (theories_mapst::iterator tit = theories_maps.begin();
         tit != theories_maps.end();
         tit++)
      {
        
        theory_functionst::const_iterator tfit = tit->second.first.find(e.id_string());
        if (tfit!=tit->second.first.end()) 
        {
          const std::list<typet> &f = tfit->second;
          for (std::list<typet>::const_iterator lit = f.begin();
               lit!=f.end();
               lit++ )
          {
            irept::named_subt::const_iterator ni = 
            lit->get_named_sub().find("annotations");
            if (ni!=lit->get_named_sub().end() &&
                ni->second.get(":assoc")=="1" &&
                lit->subtypes()[0].id()==params.subtypes()[0].id())
              {
                return lit->subtypes().back(); 
            }
          }
        }
      }
  }
      
  //std::cout << " ...found " << result.id() << std::endl;
  return smt_typecheck_exprt::search_fun(e, params);
}

/*******************************************************************\

Function: smt_typecheck_expr_generict::search_sort

  Inputs: an exprt

 Outputs: an exprt

 Purpose: searches from the current scope upwards for a definition 
          of an identifier

\*******************************************************************/

typet smt_typecheck_expr_generict::search_sort(const typet &sort)
{ 
  typet r = smt_typecheck_exprt::search_sort(sort);
  if (r.id()!="") return r;

  forall_symbols(tit, theories)
  {
    const typet &ts =
      static_cast<const typet&>(tit->second.type.find("sorts"));

    forall_subtypes(oit, ts)
      if (oit->id() == sort.id()) 
        return sort;
  }
  
  return typet(""); 
}
