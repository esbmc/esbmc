/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>

#include <set>

#include <i2string.h>

#include "boolector_prop.h"

//#define DEBUG

/*******************************************************************\

Function: boolector_propt::boolector_propt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

boolector_propt::boolector_propt(std::ostream &_out):out(_out)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  _no_variables=1;
}

/*******************************************************************\

Function: boolector_propt::~boolector_propt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

boolector_propt::~boolector_propt()
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************\

Function: boolector_propt::l_set_to

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
#if 0
void boolector_propt::l_set_to(literalt a, bool value)
{
  bvt bv;

  bv.push_back(value?a:lnot(a));

  lcnf(bv);
}
#endif
/*******************************************************************\

Function: boolector_propt::land

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_propt::land(literalt a, literalt b, literalt o)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif
  // a*b=c <==> (a + o')( b + o')(a'+b'+o)
   bvt lits;

   lits.clear();
   lits.reserve(2);
   lits.push_back(pos(a));
   lits.push_back(neg(o));
   lcnf(lits);

   lits.clear();
   lits.reserve(2);
   lits.push_back(pos(b));
   lits.push_back(neg(o));
   lcnf(lits);

   lits.clear();
   lits.reserve(3);
   lits.push_back(neg(a));
   lits.push_back(neg(b));
   lits.push_back(pos(o));
   lcnf(lits);
}

/*******************************************************************\

Function: boolector_propt::lor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_propt::lor(literalt a, literalt b, literalt o)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif
  // a+b=c <==> (a' + c)( b' + c)(a + b + c')
  bvt lits;

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(a));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  lcnf(lits);
}

/*******************************************************************\

Function: boolector_propt::lxor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_propt::lxor(literalt a, literalt b, literalt o)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif
  // a xor b = o <==> (a' + b' + o')
  //                  (a + b + o' )
  //                  (a' + b + o)
  //                  (a + b' + o)
  bvt lits;

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(neg(b));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(pos(b));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  lcnf(lits);
}

/*******************************************************************\

Function: boolector_propt::lnand

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_propt::lnand(literalt a, literalt b, literalt o)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  // a Nand b = o <==> (a + o)( b + o)(a' + b' + o')
  bvt lits;

  lits.clear();
  lits.reserve(2);
  lits.push_back(pos(a));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(2);
  lits.push_back(pos(b));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(neg(b));
  lits.push_back(neg(o));
  lcnf(lits);
}

/*******************************************************************\

Function:

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_propt::lnor(literalt a, literalt b, literalt o)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  // a Nor b = o <==> (a' + o')( b' + o')(a + b + o)
  bvt lits;

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(a));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(b));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(pos(b));
  lits.push_back(pos(o));
  lcnf(lits);
}

/*******************************************************************\

Function: boolector_propt::lequal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_propt::lequal(literalt a, literalt b, literalt o)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  lxor(a, b, lnot(o));
}

/*******************************************************************\

Function: boolector_propt::limplies

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_propt::limplies(literalt a, literalt b, literalt o)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  lor(lnot(a), b, o);
}

/*******************************************************************\

Function: boolector_propt::land

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::land(const bvt &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  literalt l=new_variable();
  u_int size=bv.size()+1;
  BtorExp *args[size], *result, *formula;

  for(unsigned int i=0; i<bv.size(); i++)
  {
	args[i] = boolector_literal(bv[i]);

    if (i==1)
      result = boolector_and(boolector_ctx, args[0], args[1]);
    else if (i>1)
      result = boolector_and(boolector_ctx, result, args[i]);
  }

  formula = boolector_iff(boolector_ctx, boolector_literal(l), result);
  boolector_assert(boolector_ctx, formula);

  if (btor)
    assumpt.push_back(formula);


#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  return l;
}

/*******************************************************************\

Function: boolector_propt::lor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::lor(const bvt &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  literalt l=new_variable();
  u_int size=bv.size()+1;
  BtorExp *args[size], *result, *formula;

  for(unsigned int i=0; i<bv.size(); i++)
  {
	args[i] = boolector_literal(bv[i]);

    if (i==1)
      result = boolector_or(boolector_ctx, args[0], args[1]);
    else if (i>1)
      result = boolector_or(boolector_ctx, result, args[i]);
  }

  formula = boolector_iff(boolector_ctx, boolector_literal(l), result);
  boolector_assert(boolector_ctx, formula);

  if (btor)
    assumpt.push_back(formula);

  return l;
}

/*******************************************************************\

Function: boolector_propt::lxor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::lxor(const bvt &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  if(bv.size()==0) return const_literal(false);
  if(bv.size()==1) return bv[0];

  literalt l=new_variable();
  u_int size=bv.size()+1;
  BtorExp *args[size], *result, *formula;

  for(unsigned int i=0; i<bv.size(); i++)
  {
	args[i] = boolector_literal(bv[i]);

    if (i==1)
      result = boolector_xor(boolector_ctx, args[0], args[1]);
    else if (i>1)
      result = boolector_xor(boolector_ctx, result, args[i]);
  }

  formula = boolector_iff(boolector_ctx, boolector_literal(l), result);
  boolector_assert(boolector_ctx, formula);

  if (btor)
    assumpt.push_back(formula);

  return l;

}
/*******************************************************************\

Function: boolector_propt::land

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::land(literalt a, literalt b)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "a: l" << a.var_no() << std::endl;
  std::cout << "b: l" << b.var_no() << std::endl;
#endif

  if(a==const_literal(true)) return b;
  if(b==const_literal(true)) return a;
  if(a==const_literal(false)) return const_literal(false);
  if(b==const_literal(false)) return const_literal(false);
  if(a==b) return a;

  literalt l=new_variable();
  BtorExp *result, *formula;

  result = boolector_and(boolector_ctx, boolector_literal(a), boolector_literal(b));
  formula = boolector_iff(boolector_ctx, boolector_literal(l), result);
  boolector_assert(boolector_ctx, formula);

  if (btor)
    assumpt.push_back(formula);

  return l;
}

/*******************************************************************\

Function: boolector_propt::lor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::lor(literalt a, literalt b)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "a: l" << a.var_no() << std::endl;
  std::cout << "b: l" << b.var_no() << std::endl;
#endif

  if(a==const_literal(false)) return b;
  if(b==const_literal(false)) return a;
  if(a==const_literal(true)) return const_literal(true);
  if(b==const_literal(true)) return const_literal(true);
  if(a==b) return a;

  literalt l=new_variable();
  BtorExp *result, *formula;

  result = boolector_or(boolector_ctx, boolector_literal(a), boolector_literal(b));
  formula = boolector_iff(boolector_ctx, boolector_literal(l), result);
  boolector_assert(boolector_ctx, formula);

  if (btor)
    assumpt.push_back(formula);

  return l;
}

/*******************************************************************\

Function: boolector_propt::lnot

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::lnot(literalt a)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "a: l" << a.var_no() << std::endl;
#endif

  a.invert();

  return a;
}

/*******************************************************************\

Function: boolector_propt::lxor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::lxor(literalt a, literalt b)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "a: l" << a.var_no() << std::endl;
  std::cout << "b: l" << b.var_no() << std::endl;
#endif

  if(a==const_literal(false)) return b;
  if(b==const_literal(false)) return a;
  if(a==const_literal(true)) return lnot(b);
  if(b==const_literal(true)) return lnot(a);

  literalt l=new_variable();
  BtorExp *result, *formula;

  result = boolector_xor(boolector_ctx, boolector_literal(a), boolector_literal(b));
  formula = boolector_iff(boolector_ctx, boolector_literal(l), result);
  boolector_assert(boolector_ctx, formula);

  if (btor)
    assumpt.push_back(formula);

  return l;
}

/*******************************************************************\

Function: boolector_propt::lnand

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::lnand(literalt a, literalt b)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "a: l" << a.var_no() << std::endl;
  std::cout << "b: l" << b.var_no() << std::endl;
#endif

  return lnot(land(a, b));
}

/*******************************************************************\

Function: boolector_propt::lnor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::lnor(literalt a, literalt b)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "a: l" << a.var_no() << std::endl;
  std::cout << "b: l" << b.var_no() << std::endl;
#endif

  return lnot(lor(a, b));
}

/*******************************************************************\

Function: boolector_propt::lequal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::lequal(literalt a, literalt b)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "a: l" << a.var_no() << std::endl;
  std::cout << "b: l" << b.var_no() << std::endl;
#endif

  return lnot(lxor(a, b));
}

/*******************************************************************\

Function: boolector_propt::limplies

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::limplies(literalt a, literalt b)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "a: l" << a.var_no() << std::endl;
  std::cout << "b: l" << b.var_no() << std::endl;
#endif

  return lor(lnot(a), b);
}

/*******************************************************************\

Function: boolector_propt::lselect

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::lselect(literalt a, literalt b, literalt c)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "a: l" << a.var_no() << std::endl;
  std::cout << "b: l" << b.var_no() << std::endl;
  std::cout << "c: l" << c.var_no() << std::endl;
#endif

  if(a==const_literal(true)) return b;
  if(a==const_literal(false)) return c;
  if(b==c) return b;

  literalt l=new_variable();
  BtorExp *result,*formula;

  result = boolector_cond(boolector_ctx, boolector_literal(a), boolector_literal(b), boolector_literal(c));
  formula = boolector_iff(boolector_ctx, boolector_literal(l), result);
  boolector_assert(boolector_ctx, formula);

  if (btor)
    assumpt.push_back(formula);

  return l;
}

/*******************************************************************\

Function: boolector_propt::new_variable

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolector_propt::new_variable()
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  literalt l;
  l.set(_no_variables, false);

  set_no_variables(_no_variables+1);

#ifdef DEBUG
  std::cout << "literal: l" << l.var_no() << "\n";
#endif

  return l;
}

/*******************************************************************\

Function: boolector_propt::eliminate_duplicates

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_propt::eliminate_duplicates(const bvt &bv, bvt &dest)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  std::set<literalt> s;

  dest.reserve(bv.size());

  for(bvt::const_iterator it=bv.begin(); it!=bv.end(); it++)
  {
    if(s.insert(*it).second)
      dest.push_back(*it);
  }
}

/*******************************************************************\

Function: boolector_propt::process_clause

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool boolector_propt::process_clause(const bvt &bv, bvt &dest)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  dest.clear();

  // empty clause! this is UNSAT
  if(bv.empty()) return false;

  std::set<literalt> s;

  dest.reserve(bv.size());

  for(bvt::const_iterator it=bv.begin();
      it!=bv.end();
      it++)
  {
    literalt l=*it;

    // we never use index 0
    assert(l.var_no()!=0);

    if(l.is_true())
      return true; // clause satisfied

    if(l.is_false())
      continue;

    if(l.var_no()>=_no_variables)
      std::cout << "l.var_no()=" << l.var_no() << " _no_variables=" << _no_variables << std::endl;
    assert(l.var_no()<_no_variables);

    // prevent duplicate literals
    if(s.insert(l).second)
      dest.push_back(l);

    if(s.find(lnot(l))!=s.end())
      return true; // clause satisfied
  }

  return false;
}

/*******************************************************************\

Function: boolector_propt::lcnf

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolector_propt::lcnf(const bvt &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  bvt new_bv;

  if(process_clause(bv, new_bv))
    return;

  BtorExp *lor_var, *args[new_bv.size()];
  unsigned int i=0, j=0;

  for(bvt::const_iterator it=new_bv.begin(); it!=new_bv.end(); it++, i++)
	args[i] = boolector_literal(*it);

  if (i>1)
  {
	lor_var = boolector_or(boolector_ctx, args[0], args[1]);

    for(j=2; j<i; j++)
      lor_var = boolector_or(boolector_ctx, args[j], lor_var);

    boolector_assert(boolector_ctx, lor_var);
    if (btor)
      assumpt.push_back(lor_var);

  }
  else if (i==1)
  {
	boolector_assert(boolector_ctx, args[0]);
    if (btor)
	  assumpt.push_back(args[0]);
  }

#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************\

Function: boolector_propt::convert_literal
  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

BtorExp* boolector_propt::convert_literal(unsigned l)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  std::string literal_s;

  literal_cachet::const_iterator cache_result=literal_cache.find(l);
  if(cache_result!=literal_cache.end())
  {
    //std::cout << "Cache hit on " << cache_result->first << "\n";
    return cache_result->second;
  }

  BtorExp* result;

  literal_s = "l"+i2string(l);
  //std::cout << "literal_s: " << literal_s << "\n";
  result = boolector_var(boolector_ctx, 1, literal_s.c_str());

  // insert into cache
  literal_cache.insert(std::pair<unsigned, BtorExp*>(l, result));

  return result;
}

/*******************************************************************\

Function: boolector_propt::boolector_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

BtorExp* boolector_propt::boolector_literal(literalt l)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  BtorExp *literal_l;

  if(l==const_literal(false))
    return boolector_false(boolector_ctx);
  else if(l==const_literal(true))
    return boolector_true(boolector_ctx);

#ifdef DEBUG
  std::cout << "l" << l.var_no() << std::endl;
#endif

  literal_l = convert_literal(l.var_no());

  if(l.sign())
  {
    return boolector_not(boolector_ctx, literal_l);
  }

  return literal_l;
}

/*******************************************************************\

Function: boolector_propt::prop_solve

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

propt::resultt boolector_propt::prop_solve()
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  return P_ERROR;
}

/*******************************************************************\

Function: boolector_propt::l_get

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

tvt boolector_propt::l_get(literalt a) const
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  tvt result=tvt(false);
  std::string literal;
  BtorExp *boolector_literal;
  size_t found;

  if(a.is_true())
    return tvt(true);
  else if(a.is_false())
    return tvt(false);

  literal_cachet::const_iterator cache_result=literal_cache.find(a.var_no());
  if(cache_result!=literal_cache.end())
    boolector_literal = cache_result->second;

  literal = boolector_bv_assignment(boolector_ctx, boolector_literal);

  found=literal.find("1");

  if (found!=std::string::npos)
  {
#ifdef DEBUG
	std::cout << "l" << a.var_no() << ": true" << std::endl;
#endif
    result=tvt(true);
  }
  else
  {
#ifdef DEBUG
	std::cout << "l" << a.var_no() << ": false" << std::endl;
#endif
    result=tvt(false);
  }

  if (a.sign()) result=!result;

  return result;
}
