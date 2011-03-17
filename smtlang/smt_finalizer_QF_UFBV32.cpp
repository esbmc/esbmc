/*******************************************************************\

Module: SMT-LIB Builtin Logics, Finalizer for the QF_UFBV32 logic  

Author: CM Wintersteiger

\*******************************************************************/

#include <errno.h>

#include "smt_typecheck.h"
#include "smt_typecheck_expr.h"
#include "expr2smt.h"

#include "builtin_theories.h"
#include "smt_finalizer_QF_UFBV32.h"

/*******************************************************************\

Function: smt_typecheck_expr_QF_UFBV32t::smt_typecheck_expr_QF_UFBV32t

  Inputs:

 Outputs:

 Purpose: constructor

\*******************************************************************/

smt_typecheck_expr_QF_UFBV32t::smt_typecheck_expr_QF_UFBV32t(  
  message_streamt &eh, 
  const exprt& b, 
  const symbolst &t) : 
    smt_typecheck_expr_generict(eh, b, t)
{
}

/*******************************************************************\

Function: smt_finalizer_QF_UFBV32t::finalize

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smt_finalizer_QF_UFBV32t::finalize(const exprt& benchmark)
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

  std::string theory="Int_ArraysEx";
  
  symbolst empty;
  check_double_sorts(benchmark, empty, err);
  check_double_functions(benchmark, empty, err);
  check_double_predicates(benchmark, empty, err);
  
  smt_typecheck_expr_QF_UFBV32t checker(err, benchmark, empty);

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

Function: smt_finalizer_QF_UFBV32t::check_double_sorts

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_finalizer_QF_UFBV32t::check_double_sorts(
  const exprt& benchmark, 
  const symbolst &theories,
  message_streamt &err) 
{
  const typet &sorts=
    static_cast<const typet &>(benchmark.find("sorts"));

  forall_subtypes(sit, sorts)
  {
    if(smt_theory_Fixed_Size_BitVectors32().check_sorts(ii2string(*sit)))
    {
      err.str <<
        "Sort symbol `" << ii2string(*sit) << "' defined twice"
        " in benchmark `" << ii2string(benchmark.find("name")) << "'";
      err.error();
    }
  }
}

/*******************************************************************\

Function: smt_finalizer_QF_UFBV32t::check_double_functions

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_finalizer_QF_UFBV32t::check_double_functions(
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
      smt_theory_Fixed_Size_BitVectors32().check_functions(
                  ii2string(*fit), params);
    
    if (t.id()!="")
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

Function: smt_finalizer_QF_UFBV32t::check_double_predicates

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_finalizer_QF_UFBV32t::check_double_predicates(
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
    if(smt_theory_Fixed_Size_BitVectors32().check_predicates( 
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

Function: smt_typecheck_expr_QF_UFBV32t::search_fun

  Inputs: an exprt and a list of typets

 Outputs: a typet

 Purpose: searches the benchmark and the theory for a fitting function
          (name in e.id() and signature in the params list)

\*******************************************************************/

typet smt_typecheck_expr_QF_UFBV32t::search_fun(
  const exprt &e,
  const typet &params) 
{ 
  //std::string estr = ii2string(e);
  if ((e.id()==strBVNAND || e.id()==strBVNOR) &&
      params.subtypes().size()==2)
  {
    int p1 = smt_theory_Fixed_Size_BitVectors32::extract_bv_size(
                    ii2string(params.subtypes()[0]));
    int p2 = smt_theory_Fixed_Size_BitVectors32::extract_bv_size(
                    ii2string(params.subtypes()[1]));
    if (!(p1==-1 || p2==-1 || p1!=p2 || p1<0 || p1>32))
    {
      if (p1>p2)      
        return params.subtypes()[0];
      else 
        return params.subtypes()[1];
    }
  }
  else if (e.id()==strBIT0 || e.id()==strBIT1)
  {
    return typet("BitVec[1]");
  }
  else if ((e.id()==strBVSLT || e.id()==strBVSLEQ || e.id()==strBVSGT ||
           e.id()==strBVSGEQ) &&
           params.subtypes().size()==2)
  {
    int p1 = smt_theory_Fixed_Size_BitVectors32::extract_bv_size(
                  ii2string(params.subtypes()[0]));
    int p2 = smt_theory_Fixed_Size_BitVectors32::extract_bv_size(
                  ii2string(params.subtypes()[1]));
    if (!(p1==-1 || p2==-1 || p1<0 || p1>32 || p2<0 || p2>32))
      return typet(strBOOL);
  }
  else if (e.id()==strSIGNEXTEND &&
           params.subtypes().size()==2 &&
           params.subtypes()[1].id()==strINT)
  {
    int p0 = smt_theory_Fixed_Size_BitVectors32::extract_bv_size(
                    ii2string(params.subtypes()[0]));
    if (p0>0 && p0 <=32 )
    {
      const std::string &p1str = params.subtypes()[1].get_string(strVALUE);
      errno=0;
      int p1=strtol(p1str.c_str(), NULL, 10);
      if (errno==0)
      {         
        int newsize = p0 + p1;
        if (newsize>0 && newsize <=32)
        {  
          std::stringstream s;
          s << "BitVec[" << newsize << "]";
          return typet(s.str());          
        }
      }
    }
  }
  else if ((e.id()==strSHIFTLEFT0 || e.id()==strSHIFTLEFT1 ||
            e.id()==strSHIFTRIGHT0 || e.id()==strSHIFTRIGHT1 ||
            e.id()==strREPEAT ||
            e.id()==strROTATELEFT || e.id()==strROTATERIGHT) &&
           params.subtypes().size()==2)
  {
    int p1 = smt_theory_Fixed_Size_BitVectors32::extract_bv_size(
                    ii2string(params.subtypes()[0]));
    if (p1!=-1 && params.subtypes()[1].id()==strINT)
    {
      return params.subtypes()[0];
    } 
  }  
  else if (e.id_string().compare(0, 4, "fill")==0 &&
           e.id_string().size()>4 && 
           params.subtypes().size()==1 &&
           (params.subtypes()[0].id()==strBIT0 ||
            params.subtypes()[0].id()==strBIT1))
  {
    std::string estr = e.id_string();
    size_t oinx = estr.rfind('[');
    size_t cinx = estr.rfind(']');
    if (oinx<cinx)
    {
      std::string inx = estr.substr(oinx+1, cinx-(oinx+1));
      errno=0;
      int t=strtoul(inx.c_str(), NULL, 10);
      if (errno==0 && t>0 && t<=32)
      {
        std::stringstream s;
        s << "BitVec[" << inx << "]";
        return typet(s.str());
      }
    }
  }
  else if (smt_theory_Fixed_Size_BitVectors32().check_predicates( 
                            ii2string(e), params))
    return typet(strBOOL);
  else   
  {
    typet t = 
      smt_theory_Fixed_Size_BitVectors32().check_functions( 
                            ii2string(e), params);
    if (t.id()!=emptySTR) return t; 
  }
    
  return smt_typecheck_exprt::search_fun(e, params);
}

/*******************************************************************\

Function: smt_typecheck_expr_QF_UFBV32t::search_sort

  Inputs: an exprt 

 Outputs: nothing

 Purpose: 

\*******************************************************************/

typet smt_typecheck_expr_QF_UFBV32t::search_sort(const typet& sort)
{   
  if (smt_theory_Fixed_Size_BitVectors32().check_sorts(ii2string(sort)))
    return typet(sort.id_string());
  return smt_typecheck_expr_generict::search_sort(sort);
}
