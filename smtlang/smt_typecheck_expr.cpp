/*******************************************************************\

Module: SMT-LIB Frontend, Typechecking of expressions

Author: CM Wintersteiger

\*******************************************************************/

#include <assert.h>

#include <list>
#include <vector>
#include <stack>

#include <expr.h>

#include "smt_typecheck.h"
#include "smt_typecheck_expr.h"
#include "expr2smt.h"

/*******************************************************************\

Function: smt_typecheck_exprt::smt_typecheck_exprt

  Inputs: 

 Outputs: 

 Purpose: constructor

\*******************************************************************/

smt_typecheck_exprt::smt_typecheck_exprt(
  message_streamt &eh, 
  const exprt &b, 
  const symbolst &t):
    smt_stringst(), 
    theories(t),
    benchmark(b),
    message_stream(eh)
{
  build_hashtables();
}

/*******************************************************************\

Function: smt_typecheck_exprt::typecheck_expr

  Inputs: an exprt

 Outputs: nothing

 Purpose: checks an expr and the symbols used in the expression 
          against a theory (and benchmark).

\*******************************************************************/

typet smt_typecheck_exprt::typecheck_expr(const exprt& e) 
{
  typet finalresult=typet(emptySTR);
  std::stack< std::pair<const exprt, typet> > exprstack;
  
  exprstack.push(
    std::pair<const exprt, typet >(e, typet())
  );
  
  while (!exprstack.empty()) {
    
    const exprt &e = exprstack.top().first;
    typet &res = exprstack.top().second;
    
    if ( res.subtypes().size()==0 ) {
      new_scope();
      check_pre(e);

      if (e.operands().size()>0)
      {
        exprstack.push(
          std::pair<const exprt, typet>(e.op0(), typet())
        );
      }
      else
      {
        assert(res.subtypes().size()==0);        
        if (exprstack.size()==1)
        {
          check_result(e, res, finalresult);
          exprstack.pop();
        }
        else {
          typet r;
          check_result(e, res, r);
          exprstack.pop();
          
          exprstack.top().second.subtypes().reserve(
            exprstack.top().second.subtypes().size()+1);
          exprstack.top().second.move_to_subtypes(r);
        }
        pop_scope();
      }
    }
    else if ( res.subtypes().size()>0 && 
              res.subtypes().size()<e.operands().size() )
    {
      exprstack.push(
        std::pair<const exprt, typet >(
          e.operands()[res.subtypes().size()]
          , 
          typet()
        )
      );
    }
    else
    {
      assert(res.subtypes().size()==e.operands().size());            
      if (exprstack.size()==1)
      {
        check_result(e, res, finalresult);
        exprstack.pop();
      }
      else 
      {
        typet r;              
        check_result(e, res, r);
        exprstack.pop();
        
        exprstack.top().second.subtypes().reserve(
            exprstack.top().second.subtypes().size()+1);        
        exprstack.top().second.move_to_subtypes(r);
      }
      pop_scope();
    }
  
  }
  
  return finalresult;
}

/*******************************************************************\

Function: smt_typecheck_exprt::check_result

  Inputs: an exprt and a list of typets

 Outputs: a typet

 Purpose: checks the subresults of an expr and returns the expressions
          global type.

\*******************************************************************/

void smt_typecheck_exprt::check_result(
  const exprt &e, 
  const typet &res,
  typet &output) 
{
  //std::cout << "Checking result for: " << e.id() << " (" << res.size() << ")" 
  //  << std::endl;

  if (res.subtypes().size()==0)
  {
    if (e.id()==strBOOL) {
      // this ...
      output.id(e.id());
      return;
    } else if (e.id()==strNATURAL) {
      // ... and this ...
      output.id(strINT);
      output.set(strVALUE, e.get(strVALUE));      
      return;
    } else if (e.id()==strRATIONAL) {
      // ... and this is not in the standard, but somehow
      // we must map numbers to sorts.
      output.id(strREAL);
      output.set(strVALUE, e.get(strVALUE));
      return;
    } else if (e.id()==strVAR) { 
      const exprt& varid = static_cast<const exprt&>(e.find(strIDENTIFIER));
      assert(varid.id()!=emptySTR);
      output = search_var(varid);
      if (output.id()==emptySTR)
      {
        message_stream.err_location(e.location());
        message_stream.error("Undeclared variable: '"+
                            varid.id_string()+"'.");
      }
      return;
    } else if (e.id()==strFVAR) { 
      const exprt& varid = static_cast<const exprt&>(e.find(strIDENTIFIER));
      assert(varid.id()!=emptySTR);
      output = search_fun(varid, typet());      
      if (output.id()==emptySTR)
      {
        message_stream.err_location(e.location());
        message_stream.error("Undeclared function variable: '"+
                            varid.id_string()+"'.");
      }
      return;
    } else {
      // assume it's a function or predicate without parameters
      output = search_fun(e, typet());
      if (output.id()==emptySTR) {
        message_stream.err_location(e.location());
        std::string errmsg;
        errmsg = "No function or predicate by the name '"+e.id_string();
        errmsg+= "' with signature ( nil -> *) was declared.";
        message_stream.error(errmsg);
      } else
        return;
    }
  } else { // res.subtypes().size() > 0
    //std::cout << "Check: " << e.id() << ", " << e.operands().size() << 
    //  " operands -> ";
    if (e.id()==strFORALL ||
        e.id()==strEXISTS ||
        e.id()==strLET ||
        e.id()==strFLET) 
    {
      output = res.subtypes()[0]; 
      return;
    } else if (e.id()==strIFTHENELSE || e.id()==strITE) {
      if (res.subtypes().size()!=2 && res.subtypes().size()!=3) 
      {
        message_stream.err_location(e.location());
        message_stream.error("if_then_else must have 2 or 3 arguments.");
      } 
      else if (res.subtypes()[0].id()!=strBOOL)
      {
        message_stream.err_location(e.location());
        message_stream.error("if_then_else must have a boolean guard.");
      } 
      else if (res.subtypes().size()==3 && 
               res.subtypes()[1].id()!=res.subtypes()[2].id())
      {
        message_stream.err_location(e.location());
        std::string errmsg;
        message_stream.error("Conflicting types for if_then_else arguments: "+
                            res.subtypes()[1].id_string() + " vs. " + 
                            res.subtypes()[2].id_string() + ".");
      } else {
        output = res.subtypes()[1];
        return;
      }
    } else {
      // we should have some function here.
      output = search_fun(e, res);
      if (output.id()==emptySTR) {
        message_stream.err_location(e.location());
        std::string errmsg;
        errmsg="No function or predicate by the name '"+e.id_string();
        errmsg+="' with signature (";
        std::string t;        
        for (unsigned i=0; i<res.subtypes().size()-1; i++)
        {          
          type2smt(res.subtypes()[i], t);          
          errmsg+= ((t=="")?"?":t) + " x ";
        }
        type2smt(res.subtypes().back(), t);
        errmsg+= ((t=="")?"?":t) + " -> *) was declared.";
        message_stream.error(errmsg);
      } else
        return;
    }
    //std::cout << result.id() << std::endl;
  }

  output.id(emptySTR);
  return;
}

/*******************************************************************\

Function: smt_typecheck_exprt::check_pre

  Inputs: an exprt 

 Outputs: nothing

 Purpose: 

\*******************************************************************/

void smt_typecheck_exprt::check_pre(const exprt &e) 
{
  if (e.id()==strFORALL ||
      e.id()==strEXISTS) 
  {
    const exprt& qv = static_cast<const exprt&>(e.find(strQVARS));
    forall_operands(it, qv) {
      irept::named_subt::const_iterator ni = 
        it->get_named_sub().find(strIDENTIFIER); // could be 'nil'
      assert (ni!=it->get_named_sub().end());
      irept varname = ni->second;
      current_scope().ids[varname.id_string()]=*it;
    }
    assert(e.operands().size()==1);
  }
  else if(e.id()==strLET)
  {
    const exprt& vars = static_cast<const exprt&>(e.find(strVARIABLES));
    const exprt& terms = static_cast<const exprt&>(e.find(strVTERMS));    
    assert(vars.operands().size()==terms.operands().size());
    
    // the order of these declarations is reversed! (see parser)
    for ( exprt::operandst::const_reverse_iterator
          vit = vars.operands().rbegin(),
          tit = terms.operands().rbegin();
          vit != vars.operands().rend() && tit != terms.operands().rend();
          vit++, tit++)
    {
      exprt var = *vit;
      exprt term = *tit;
      // TODO: get rid of the recursion!
      var.add(strTYPE).add(strTYPE) = typecheck_expr(term);
      // its okay to overwrite existing ones here
      irept::named_subt::const_iterator ni = 
        vit->get_named_sub().find(strIDENTIFIER); // could be 'nil'
      assert (ni!=vit->get_named_sub().end());
      irept varname = ni->second;      
      current_scope().ids[varname.id_string()]=var;
    }
  }
  else if (e.id()==strFLET)
  {
    const exprt& vars = static_cast<const exprt&>(e.find(strVARIABLES));
    const exprt& formulas = static_cast<const exprt&>(e.find(strVFORMULAS));    
    assert(vars.operands().size()==formulas.operands().size());
    
    // the order of these declarations is reversed! (see parser)
    for ( exprt::operandst::const_reverse_iterator 
          vit = vars.operands().rbegin(),
          fit = formulas.operands().rbegin();
          vit != vars.operands().rend() && fit != formulas.operands().rend();
          vit++, fit++)
    {
      exprt var = *vit;
      exprt formula = *fit;
      // TODO: get rid of the recursion!
      var.add(strRETURNTYPE) = typecheck_expr(formula);
      irept::named_subt::const_iterator ni = 
        var.get_named_sub().find(strIDENTIFIER); // could be 'nil'
      assert (ni!=vit->get_named_sub().end());
      irept varname = ni->second;
      // recklessly overwrite it, it won't harm.
      current_scope().fdefs[varname.id_string()]=var;
    }
  }
}

/*******************************************************************\

Function: smt_typecheck_exprt::build_hashtables

  Inputs: none

 Outputs: nothing

 Purpose: builds the hashtables for function and predicate symbols

\*******************************************************************/

void smt_typecheck_exprt::build_hashtables(void) 
{
  forall_symbols(it, theories)
  {
    const typet &fs =
        static_cast<const typet&>(it->second.type.find(strFUNCTIONS));
    const typet &ps =
        static_cast<const typet&>(it->second.type.find(strPREDICATES));
        
    theory_functionst &funs=theories_maps[it->second.base_name].first;
    theory_predicatest &preds=theories_maps[it->second.base_name].second;

    forall_subtypes(fit, fs)
      funs[fit->id_string()].push_back(*fit);
    forall_subtypes(pit, ps)
      preds[pit->id_string()].push_back(*pit);
  }
  
  const typet &bfuns = 
    static_cast<const typet&>(benchmark.find(strFUNCTIONS));
  forall_subtypes(it, bfuns)
    benchmark_functions[it->id_string()].push_back(*it);
  
  const typet& bpreds =
      static_cast<const typet&>(benchmark.find(strPREDICATES));
  forall_subtypes(it, bpreds)
    benchmark_predicates[it->id_string()].push_back(*it);
}

/*******************************************************************\

Function: smt_typecheck_exprt::search_var

  Inputs: an exprt

 Outputs: an exprt

 Purpose: searches from the current scope upwards for a definition 
          of an identifier

\*******************************************************************/

typet smt_typecheck_exprt::search_var(const exprt &e) 
{
  //std::cout << "Searching variable: " << e.id() << std::endl;
  
  for(scopest::const_reverse_iterator it = scopes.rbegin();
      it!=(scopest::const_reverse_iterator)scopes.rend();
      it++)
  {     
    scopet::idst::const_iterator iit = it->ids.find(e.id());
    
    if (iit!=it->ids.end()){
      const exprt &var = iit->second;
      
      if (var.id()==strVAR) { // found      
        const typet& sort =
          static_cast<const typet&>(var.find(strTYPE).find(strTYPE));
  
        typet r = search_sort(sort);
        if (r.id()!=emptySTR) 
          return r;
        else
        {
          message_stream.err_location(var.location());
          message_stream.error("Unknown sort used: `"+sort.id_string()+"'");
        }
      }
    }
  }  
  
  return typet(emptySTR);
}


/*******************************************************************\

Function: smt_typecheck_exprt::search_fun

  Inputs: an exprt and a list of typets

 Outputs: a typet

 Purpose: searches the benchmark and the theory for a fitting function
          (name in e.id() and signature in the params list)

\*******************************************************************/

typet smt_typecheck_exprt::search_fun(
  const exprt &e,
  const typet &params) 
{
  if (params.subtypes().size()==0)
  { // check flet definitions    
    for(scopest::const_reverse_iterator
        it=scopes.rbegin();
        it!=(scopest::const_reverse_iterator)scopes.rend();
        it++)
    {      
      scopet::fdefst::const_iterator fit = it->fdefs.find(e.id());
      
      if (fit!=it->fdefs.end())
      {
        const exprt &fvar = fit->second;
        if (fvar.id()==strFVAR) // found
        {
          return typet(fvar.find(strRETURNTYPE).id());
        }
      }
    }
  }
  
  typet paramsplus = params;
  paramsplus.copy_to_subtypes(typet());
  
  { // check benchmark-functions
    //std::cout << "Checking benchmark functions" << std::endl;
    benchmark_functionst::const_iterator it = 
      benchmark_functions.find(e.id_string());

    if (it!=benchmark_functions.end()) {
      const std::list<typet> &funs=it->second;
      
      for (std::list<typet>::const_iterator lit = funs.begin();
           lit!=funs.end();
           lit++ )
      {                
        if (smt_typecheckt::signature_eq_except_last(*lit, paramsplus)) 
        {          
          return lit->subtypes().back(); 
        }
      }
    }
  }
  
  { // check benchmark-predicates
    // std::cout << "Checking benchmark predicates" << std::endl;
    benchmark_predicatest::const_iterator it = 
      benchmark_predicates.find(e.id_string());

    if (it!=benchmark_predicates.end()) {
      const std::list<typet> &funs=it->second;
    
      for (std::list<typet>::const_iterator lit = funs.begin();
           lit!=funs.end();
           lit++ )
      {
        if (smt_typecheckt::signature_eq(*lit, params)) 
        {
          return typet(strBOOL);
        }
      }
    }
  }
  
  if (params.subtypes().size()>0)
  { // check the builtin boolean connectives
    bool allbools = true;

    for (unsigned i=0; i<params.subtypes().size(); i++)
    {
      if (params.subtypes()[i].id()!=strBOOL) {
        allbools=false; break;
      }
    }

    if (allbools)
    {
      if (e.id()==strAND ||
          e.id()==strOR ||
          e.id()==strXOR || 
          e.id()==strIFF ||
          (e.id()==strNOT && params.subtypes().size()==1) ||
          (  (e.id()==strIMPLIES || 
              e.id()==strIMPL) /* nonstandard! */
            && params.subtypes().size()==2)          
          ) 
      {
        return typet(strBOOL);
      }
    }
  }
  
  if ( (e.id()==strArEQUAL && params.subtypes().size()>0) ||
       (e.id()==strDISTINCT && params.subtypes().size()>1))
  { // check the builtin equality symbol
    bool allthesame=true;
    typet last = params.subtypes()[0];

    for (unsigned i=1; i<params.subtypes().size(); i++)
    {
      if (params.subtypes()[i].id()!=last.id())
      {
        allthesame=false; break;
      }
      else
      {
        last.id(params.subtypes()[i].id());
      }
    }

    if (allthesame)
      return typet(strBOOL);
  }
  
  if (e.id()==strITE && params.subtypes().size()==3)
  { // check the builtin ite function
    if (params.subtypes()[0].id()==strBOOL &&
        params.subtypes()[1].id()==params.subtypes()[2].id())
      return params.subtypes()[1];
  }
  
  // if we didnt't find one until now, let's try for :assoc marked ones
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
    {
      // in the benchmark-functions
      benchmark_functionst::const_iterator it = 
        benchmark_functions.find(e.id_string());
        
      if (it!=benchmark_functions.end()) {
        const std::list<typet> &funs=it->second;
        
        for (std::list<typet>::const_iterator lit = funs.begin();
             lit!=funs.end();
             lit++ )
        { 
          irept::named_subt::const_iterator ni = 
            lit->get_named_sub().find(strANNOTATIONS);
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
  
  return typet(emptySTR);
}

/*******************************************************************\

Function: smt_typecheck_exprt::search_sort

  Inputs: an exprt

 Outputs: an exprt

 Purpose: searches from the current scope upwards for a definition 
          of an identifier

\*******************************************************************/

typet smt_typecheck_exprt::search_sort(const typet &sort)
{ 

  const typet& sorts =
    static_cast<const typet&>(benchmark.find(strSORTS));

  forall_subtypes(sit, sorts)
  {
    if (sit->id() == sort.id()) 
      return sort;
  }
  
  return typet(emptySTR);
}
