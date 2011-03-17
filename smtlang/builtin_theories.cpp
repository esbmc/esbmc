/*******************************************************************\

Module: SMT-LIB Builtin Theories  

Author: CM Wintersteiger

\*******************************************************************/

#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <i2string.h>

#include "expr2smt.h"
#include "builtin_theories.h"

// reusable definitions of sorts, predicates and functions
static const char* sort_Int = "Int";
static const char* sort_Real = "Real";
static const char* sort_Array = "Array";
static const char* sort_Array1 = "Array1";
static const char* sort_Array2 = "Array2";
static const char* sort_BitVec = "BitVec";

static const char* predicate_LT_Int[] = {"<", sort_Int, sort_Int, NULL};
static const char* predicate_GT_Int[] = {">", sort_Int, sort_Int, NULL};
static const char* predicate_LTE_Int[] = {"<=", sort_Int, sort_Int, NULL};
static const char* predicate_GTE_Int[] = {">=", sort_Int, sort_Int, NULL};

static const char* predicate_LT_Real[] = {"<", sort_Real, sort_Real, NULL};
static const char* predicate_GT_Real[] = {">", sort_Real, sort_Real, NULL};
static const char* predicate_LTE_Real[] = {"<=", sort_Real, sort_Real, NULL};
static const char* predicate_GTE_Real[] = {">=", sort_Real, sort_Real, NULL};

static const char* function_0_Int[] = {"0", sort_Int, NULL};
static const char* function_1_Int[] = {"1", sort_Int, NULL};
static const char* function_Neg_Int[] = {"~", sort_Int, sort_Int, NULL};
static const char* function_UMinus_Int[] = {"-", sort_Int, sort_Int, NULL};
static const char* function_Minus_Int[] = 
  {"-", sort_Int, sort_Int, sort_Int, NULL};
static const char* function_Plus_Int[] = 
  {"+", sort_Int, sort_Int, sort_Int, NULL};
static const char* function_Times_Int[] = 
  {"*", sort_Int, sort_Int, sort_Int, NULL};
  
static const char* function_0_Real[] = {"0.0", sort_Real, NULL};
static const char* function_1_Real[] = {"1.0", sort_Real, NULL};
static const char* function_UMinus_Real[] = {"-", sort_Real, sort_Real, NULL};
static const char* function_Minus_Real[] = 
  {"-", sort_Real, sort_Real, sort_Real, NULL};
static const char* function_Plus_Real[] = 
  {"+", sort_Real, sort_Real, sort_Real, NULL};
static const char* function_Times_Real[] = 
  {"*", sort_Real, sort_Real, sort_Real, NULL};


static const char* function_select_Int[] = 
  {"select", sort_Array, sort_Int, sort_Int, NULL};
static const char* function_store_Int[] = 
  {"store", sort_Array, sort_Int, sort_Int, sort_Array, NULL};
  
static const char* function_select_Array1_Int_PLUS[] = 
  {"select", sort_Array1, sort_Int, sort_Int, NULL};
static const char* function_store_Array1_Int_PLUS[] = 
  {"store", sort_Array1, sort_Int, sort_Int, sort_Array1, NULL};
  
static const char* function_select_Int_Real[] = 
  {"select", sort_Array1, sort_Int, sort_Real, NULL};
static const char* function_store_Int_Real[] = 
  {"store", sort_Array1, sort_Int, sort_Real, sort_Array1, NULL};

static const char* function_select_Array2_Int_PLUS[] = 
  {"select", sort_Array2, sort_Int, sort_Int, NULL};
static const char* function_store_Array2_Int_PLUS[] = 
  {"store", sort_Array2, sort_Int, sort_Int, sort_Array2, NULL};

static const char* function_select_Int_Array1[] = 
  {"select", sort_Array2, sort_Int, sort_Array1, NULL};
static const char* function_store_Int_Array1[] = 
  {"store", sort_Array2, sort_Int, sort_Array1, sort_Array2, NULL};  

// Theory Ints

const char* smt_theory_Ints::sorts[] = {
  sort_Int, 
  NULL
};

const char** smt_theory_Ints::predicates[] = {
  predicate_LTE_Int,
  predicate_LT_Int,
  predicate_GTE_Int,
  predicate_GT_Int,
  NULL       
};
  
const char** smt_theory_Ints::functions[] = {
  function_0_Int,
  function_1_Int,
  function_Neg_Int,
  function_Minus_Int,
  function_Plus_Int,
  function_Times_Int,
  NULL       
};

// Theory Reals

const char* smt_theory_Reals::sorts[] = {
  sort_Real, 
  NULL
};

const char** smt_theory_Reals::predicates[] = {
  predicate_LTE_Real,
  predicate_LT_Real,
  predicate_GTE_Real,
  predicate_GT_Real,
  NULL       
};
  
const char** smt_theory_Reals::functions[] = {
  function_0_Real,
  function_1_Real,
  function_UMinus_Real,
  function_Minus_Real,
  function_Plus_Real,
  function_Times_Real,
  NULL       
};

// Theory Int_Arrays

const char* smt_theory_Int_Arrays::sorts[] = {
  sort_Int, 
  sort_Array,
  NULL
};

const char** smt_theory_Int_Arrays::predicates[] = {
  predicate_LTE_Int,
  predicate_LT_Int,
  predicate_GTE_Int,
  predicate_GT_Int,
  NULL       
};
  
const char** smt_theory_Int_Arrays::functions[] = {
  function_0_Int,
  function_1_Int,
  function_Neg_Int,
  function_Minus_Int,
  function_Plus_Int,
  function_Times_Int,
  function_select_Int,
  function_store_Int,
  NULL       
};

// Theory Int_Int_Real_Array_ArraysEx


const char* smt_theory_Int_Int_Real_Array_ArraysEx::sorts[] = {
  sort_Int,
  sort_Real,   
  sort_Array1,
  sort_Array2,
  NULL
};

const char** smt_theory_Int_Int_Real_Array_ArraysEx::predicates[] = {
  predicate_LTE_Int,
  predicate_LT_Int,
  predicate_GTE_Int,
  predicate_GT_Int,
  predicate_LTE_Real,
  predicate_LT_Real,
  predicate_GTE_Real,
  predicate_GT_Real,
  NULL       
};
  
const char** smt_theory_Int_Int_Real_Array_ArraysEx::functions[] = {
  function_0_Int,
  function_1_Int,
  function_UMinus_Int,
  function_Minus_Int,
  function_Plus_Int,
  function_Times_Int,
  function_0_Real,
  function_1_Real,
  function_UMinus_Real,
  function_Minus_Real,
  function_Plus_Real,
  function_Times_Real,
  function_select_Int_Real,
  function_store_Int_Real,
  function_select_Int_Array1,
  function_store_Int_Array1,
  function_select_Array1_Int_PLUS, /* Addition! */
  function_store_Array1_Int_PLUS, /* Addition! */
  function_select_Array2_Int_PLUS, /* Addition! */
  function_store_Array2_Int_PLUS, /* Addition! */
  NULL       
};


// Theory smt_theory_Fixed_Size_BitVectors32

const char* smt_theory_Fixed_Size_BitVectors32::sorts[] = {
  NULL // Special handling!
};

const char** smt_theory_Fixed_Size_BitVectors32::predicates[] = {
  NULL // Special handling!
};

const char** smt_theory_Fixed_Size_BitVectors32::functions[] = {
  NULL // Special handling!
};

/*******************************************************************\

Function: smt_check_predicates

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_theory::check_predicates( 
  const std::string& e, 
  const typet& params)
{  
  for ( const char*** cur = get_predicates();
        *cur!=NULL;
        cur++)
  {
    const char* pname = **cur;
    if ( e==pname )             
    {
      unsigned i;
      for (i=1; (*cur)[i]!=NULL; i++)
      {
        if (ii2string(params.subtypes()[i-1])!=(*cur)[i])
          break;
      }
      if (i==(params.subtypes().size()+1) && (*cur)[i]==NULL) // all checked
        return true;
    }
  }
  return false;
}

/*******************************************************************\

Function: smt_check_functions

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

typet smt_theory::check_functions(
  const std::string& e, 
  const typet& params)
{  
  for ( const char*** cur = get_functions();
        *cur!=NULL;
        cur++)
  {
    const char* pname = **cur;
    if ( e==pname )
    {
      bool isequal = true;
      unsigned i;
      for (i=1; i<=params.subtypes().size(); i++)
      {        
        if ((*cur)[i]==NULL || 
            ii2string(params.subtypes()[i-1])!=(*cur)[i])
        {
          isequal=false; // signature not equal!
          break;
        }
      }
      // all checked
      if (isequal)
      {
        if ((*cur)[i]!=NULL)
        {   
          typet next = typet((*cur)[i]);
          if ((*cur)[i+1]==NULL) // this must be the last in the list!
            return next; // found!
          // else it didnt fit
        }
        // else we don't have a return value -- search on
      }
    }
  }
  return typet(""); // no function by that name found
}

/*******************************************************************\

Function: smt_check_sorts

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_theory::check_sorts(const std::string& sort)
{  
  for ( const char** cur = get_sorts();
        *cur!=NULL;
        cur++)
  {    
    if ( sort == *cur )             
        return true;
  }
  return false;
}

/*******************************************************************\

Function: smt_theory_Fixed_Size_BitVectors32::smt_check_predicates

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_theory_Fixed_Size_BitVectors32::check_predicates(
  const std::string& e, 
  const typet& params)
{
  if ((e=="<" || e=="<=" || e==">=" || e==">" || e=="=" ||
       e=="bvlt" || e=="bvleq" || e=="bvgeq" || e=="bvgt") &&
       params.subtypes().size()==2)
  {
    int p1 = extract_bv_size(ii2string(params.subtypes()[0]));
    int p2 = extract_bv_size(ii2string(params.subtypes()[1]));    
    if (p1!=p2 || p1==-1 || p1<0 || p1>32)
      return false;
    else
      return true;
  }
  return smt_theory::check_predicates(e, params);
}

/*******************************************************************\

Function: smt_theory_Fixed_Size_BitVectors32::smt_check_functions

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

typet smt_theory_Fixed_Size_BitVectors32::check_functions(
  const std::string& e, 
  const typet& params)
{
  if (e.compare(0, 2, "bv")==0 &&
       e.size()>2 && 
       params.subtypes().size()==0) 
  {
    errno=0; // *NONSTANDARD* the theory just talks about bv0 and bv1!
    int t = strtoul(e.c_str(), NULL, 10);
    if (errno==0 && t>=0)
      return typet("BitVec[32]");
  }
  else if (e=="concat" &&
           params.subtypes().size()==2)
  {
    int p1 = extract_bv_size(ii2string(params.subtypes()[0]));
    int p2 = extract_bv_size(ii2string(params.subtypes()[1]));
    
    if (!(p1==-1 || p2==-1 || (p1+p2)>32 || (p1+p2)<0))
    {
      std::string t(sort_BitVec);
      t += "[" + i2string(p1+p2) + "]";
      return typet(t);
    }
  }
  else if (e.compare(0, 7, "extract")==0)
  {
    std::pair<int, int> rng = extract_bv_range(e);
    if (rng.first==-1 || rng.second==-1 || 
        rng.first>32 || rng.first<0 || rng.first<rng.second)
      return typet("");
    else
    {
      std::string t(sort_BitVec);
      t += "[" + i2string(rng.first-rng.second+1) + "]";
      return typet(t);
    }
  }  
  else if ((e=="bvnot" || e=="bvneg") && 
            params.subtypes().size()==1)
  {
    int p1 = extract_bv_size(ii2string(params.subtypes()[0]));
    if (!(p1==-1 || p1<0 || p1>32))
      return params.subtypes()[0];
  }
  else if ((e=="bvand" || e=="bvor" || e=="bvxor" || 
            e=="bvsub" || e=="bvadd" || e=="bvmul") &&
            params.subtypes().size()==2)
  {
    int p1 = extract_bv_size(ii2string(params.subtypes()[0]));
    int p2 = extract_bv_size(ii2string(params.subtypes()[1]));
    if (!(p1==-1 || p2==-1 || p1<0 || p1>32 || p2<0 || p2>32))      
    {
      if (p1>p2)      
        return params.subtypes()[0];
      else 
        return params.subtypes()[1];
    }
  }
  else if (e=="ite" &&
           params.subtypes().size()==3)
  {
    if (params.subtypes()[0].id()=="bool")
    {
      int p1 = extract_bv_size(ii2string(params.subtypes()[1]));
      int p2 = extract_bv_size(ii2string(params.subtypes()[2]));
      if (!(p1==-1 || p2==-1 || p1<0 || p1>32 || p2<0 || p2>32))
      {
        if (p1>p2)      
          return params.subtypes()[0];
        else 
          return params.subtypes()[1];
      }
    }
  }

  return smt_theory::check_functions(e, params);
}

/*******************************************************************\

Function: smt_theory_Fixed_Size_BitVectors32::smt_check_sorts

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool smt_theory_Fixed_Size_BitVectors32::check_sorts(const std::string& sort)
{
  if (extract_bv_size(sort)!=-1) return true;
  return smt_theory::check_sorts(sort);
}

/*******************************************************************\

Function: smt_theory_Fixed_Size_BitVectors32::extract_bv_size

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

int smt_theory_Fixed_Size_BitVectors32::extract_bv_size( const std::string& s )
{
  static unsigned bvstrlen = strlen(sort_BitVec);
  if (s.compare(0, bvstrlen, sort_BitVec)==0 &&
      s[bvstrlen] == '[' &&
      s.size()>bvstrlen && 
      s[s.size()-1]==']')
  {
    errno=0;
    unsigned t=strtoul(s.substr(bvstrlen+1, s.size()-bvstrlen).c_str(),
                        NULL, 10);
    if (errno==0 && t<=32)
      return t;
  }
  return -1;
}

/*******************************************************************\

Function: smt_theory_Fixed_Size_BitVectors32::extract_bv_range

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

std::pair<int, int> smt_theory_Fixed_Size_BitVectors32::extract_bv_range( 
  const std::string& s )
{
  size_t oinx = s.rfind('[');
  size_t cinx = s.rfind(']');
  if (oinx!=std::string::npos &&
      cinx!=std::string::npos && 
      oinx<cinx )
  {
    const std::string rng = s.substr(oinx+1, cinx-(oinx+1));
    size_t colinx = rng.find(':');
    const std::string l = rng.substr(0, colinx);
    const std::string r = rng.substr(colinx+1);
 
    errno=0;
    int li=strtoul(l.c_str(), NULL, 10);
    if (errno!=0 || li<0 || li>32)
      return std::pair<int, int>(-1, -1);
    
    errno=0;
    int ri=strtoul(r.c_str(), NULL, 10);
    if (errno!=0 || ri<0 || ri>32)
      return std::pair<int, int>(-1, -1);
    
    if (li>=ri && ((li-ri+1)<=32))
      return std::pair<int, int>(li, ri);
  }
  return std::pair<int, int>(-1, -1);
}
