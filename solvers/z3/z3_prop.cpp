/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>
#include <malloc.h>
#include <set>
#include <i2string.h>

#include "z3_prop.h"

//#define DEBUG

/*******************************************************************\

Function: z3_propt::z3_propt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

z3_propt::z3_propt(std::ostream &_out, bool uw):out(_out)
{
  _no_variables=1;
  this->uw = uw;
}

/*******************************************************************\

Function: z3_propt::~z3_propt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

z3_propt::~z3_propt()
{
}

/*******************************************************************\

Function: z3_propt::land

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_propt::land(literalt a, literalt b, literalt o)
{
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

Function: z3_propt::lor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_propt::lor(literalt a, literalt b, literalt o)
{
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

Function: z3_propt::lxor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_propt::lxor(literalt a, literalt b, literalt o)
{
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

Function: z3_propt::lnand

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_propt::lnand(literalt a, literalt b, literalt o)
{
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

void z3_propt::lnor(literalt a, literalt b, literalt o)
{
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

Function: z3_propt::lequal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_propt::lequal(literalt a, literalt b, literalt o)
{
  lxor(a, b, lnot(o));
}

/*******************************************************************\

Function: z3_propt::limplies

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_propt::limplies(literalt a, literalt b, literalt o)
{
  lor(lnot(a), b, o);
}

/*******************************************************************\

Function: z3_propt::land

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::land(const bvt &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  literalt l=new_variable();
  uint size=bv.size();
  Z3_ast *args = (Z3_ast*)alloca(size * sizeof(Z3_ast));
  Z3_ast result, formula;

  for(unsigned int i=0; i<bv.size(); i++)
	args[i] = z3_literal(bv[i]);

  result = Z3_mk_and(z3_ctx, bv.size(), args);
  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;
}

/*******************************************************************\

Function: z3_propt::lor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::lor(const bvt &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  literalt l=new_variable();
  uint size=bv.size();
  Z3_ast *args = (Z3_ast*)alloca(size * sizeof(Z3_ast));
  Z3_ast result, formula;

  for(unsigned int i=0; i<bv.size(); i++)
	args[i] = z3_literal(bv[i]);

  result = Z3_mk_or(z3_ctx, bv.size(), args);

  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;
}

/*******************************************************************\

Function: z3_propt::lxor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::lxor(const bvt &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  if(bv.size()==0) return const_literal(false);
  if(bv.size()==1) return bv[0];

  literalt l=new_variable();
  uint size=bv.size();
  Z3_ast *args = (Z3_ast *)alloca(size * sizeof(Z3_ast));
  Z3_ast result=0, formula;

  for(unsigned int i=0; i<bv.size(); i++)
  {
	args[i] = z3_literal(bv[i]);

    if (i==1)
      result = Z3_mk_xor(z3_ctx, args[0], args[1]);
    else if (i>1)
      result = Z3_mk_xor(z3_ctx, result, args[i]);
  }

  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;

}
/*******************************************************************\

Function: z3_propt::land

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::land(literalt a, literalt b)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif
#if 1
  if(a==const_literal(true)) return b;
  if(b==const_literal(true)) return a;
  if(a==const_literal(false)) return const_literal(false);
  if(b==const_literal(false)) return const_literal(false);
  if(a==b) return a;
#endif
  literalt l=new_variable();
  Z3_ast result, operand[2], formula;

  operand[0] = z3_literal(a);
  operand[1] = z3_literal(b);
  result = Z3_mk_and(z3_ctx, 2, operand);
  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;

}

/*******************************************************************\

Function: z3_propt::lor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::lor(literalt a, literalt b)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif
#if 1
  if(a==const_literal(false)) return b;
  if(b==const_literal(false)) return a;
  if(a==const_literal(true)) return const_literal(true);
  if(b==const_literal(true)) return const_literal(true);
  if(a==b) return a;
#endif
  literalt l=new_variable();
  Z3_ast result, operand[2], formula;

  operand[0] = z3_literal(a);
  operand[1] = z3_literal(b);
  result = Z3_mk_or(z3_ctx, 2, operand);
  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  return l;

}

/*******************************************************************\

Function: z3_propt::lnot

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::lnot(literalt a)
{
  a.invert();

  return a;
}

/*******************************************************************\

Function: z3_propt::lxor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::lxor(literalt a, literalt b)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif
#if 1
  if(a==const_literal(false)) return b;
  if(b==const_literal(false)) return a;
  if(a==const_literal(true)) return lnot(b);
  if(b==const_literal(true)) return lnot(a);
#endif
  literalt l=new_variable();
  Z3_ast result, operand[2], formula;

  operand[0] = z3_literal(a);
  operand[1] = z3_literal(b);
  result = Z3_mk_xor(z3_ctx, operand[0], operand[1]);
  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;

}

/*******************************************************************\

Function: z3_propt::lnand

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::lnand(literalt a, literalt b)
{
  return lnot(land(a, b));
}

/*******************************************************************\

Function: z3_propt::lnor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::lnor(literalt a, literalt b)
{
  return lnot(lor(a, b));
}

/*******************************************************************\

Function: z3_propt::lequal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::lequal(literalt a, literalt b)
{
  return lnot(lxor(a, b));
}

/*******************************************************************\

Function: z3_propt::limplies

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::limplies(literalt a, literalt b)
{
  return lor(lnot(a), b);
}

/*******************************************************************\

Function: z3_propt::lselect

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::lselect(literalt a, literalt b, literalt c)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif
#if 1
  if(a==const_literal(true)) return b;
  if(a==const_literal(false)) return c;
  if(b==c) return b;
#endif
  literalt l=new_variable();
  Z3_ast result, formula;

  result = Z3_mk_ite(z3_ctx, z3_literal(a), z3_literal(b), z3_literal(c));
  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;
}

/*******************************************************************\

Function: z3_propt::new_variable

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt z3_propt::new_variable()
{
  literalt l;

  l.set(_no_variables, false);

  set_no_variables(_no_variables+1);

#ifdef DEBUG
  std::cout << "new literal: l" << l.var_no() << "\n";
#endif

  return l;
}

/*******************************************************************\

Function: z3_propt::eliminate_duplicates

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_propt::eliminate_duplicates(const bvt &bv, bvt &dest)
{
  std::set<literalt> s;

  dest.reserve(bv.size());

  for(bvt::const_iterator it=bv.begin(); it!=bv.end(); it++)
  {
    if(s.insert(*it).second)
      dest.push_back(*it);
  }
}

/*******************************************************************\

Function: z3_propt::process_clause

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool z3_propt::process_clause(const bvt &bv, bvt &dest)
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

Function: z3_propt::lcnf

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void z3_propt::lcnf(const bvt &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  bvt new_bv;

  if(process_clause(bv, new_bv))
    return;

  if (new_bv.size()==0)
    return;

  Z3_ast lor_var, *args = (Z3_ast*)alloca(new_bv.size() * sizeof(Z3_ast));
  unsigned int i=0;

  for(bvt::const_iterator it=new_bv.begin(); it!=new_bv.end(); it++, i++)
	args[i] = z3_literal(*it);

  if (i>1)
  {
    lor_var = Z3_mk_or(z3_ctx, i, args);
    assert_formula(lor_var);
  }
  else
  {
    assert_formula(args[0]);
  }
}

/*******************************************************************\

Function: z3_propt::z3_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

Z3_ast z3_propt::z3_literal(literalt l)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  Z3_ast literal_l;
  std::string literal_s;

  if(l==const_literal(false))
    return Z3_mk_false(z3_ctx);
  else if(l==const_literal(true))
    return Z3_mk_true(z3_ctx);

  literal_s = "l"+i2string(l.var_no());
  literal_l = z3_api.mk_bool_var(z3_ctx, literal_s.c_str());

#ifdef DEBUG
  std::cout << "literal_s: " << literal_s << "\n";
#endif

  if(l.sign())
  {
	//std::cout << "not " << literal_s << "\n";
    return Z3_mk_not(z3_ctx, literal_l);
  }

  return literal_l;
}

/*******************************************************************\

Function: z3_propt::prop_solve

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

propt::resultt z3_propt::prop_solve()
{
  return P_ERROR;
}

/*******************************************************************\

Function: z3_propt::l_get

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

tvt z3_propt::l_get(literalt a) const
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  tvt result=tvt(tvt::TV_ASSUME);
  std::string literal;
  Z3_ast z3_literal;
  size_t found, found2;

  //std::cout << "a.var_no(): " << a.var_no() << "\n";

  if(a.is_true())
  {
	//std::cout << "true" << "\n";
    return tvt(true);
  }
  else if(a.is_false())
  {
	//std::cout << "false" << "\n";
	return tvt(false);
  }

  unsigned v=a.var_no();
  if(v>=map_prop_vars.size())
  {
	//std::cout << "TV_UNKNOWN " << "\n";
    return tvt(tvt::TV_UNKNOWN);
  }

  literal = "l"+i2string(a.var_no());

  map_prop_varst::const_iterator cache_result=map_prop_vars.find(literal.c_str());

  //std::cout << "literal1: " << literal << "\n";

  if(cache_result!=map_prop_vars.end())
  {
    //std::cout << "Cache hit on " << cache_result->first << "\n";
	z3_literal = cache_result->second;
    Z3_app app = Z3_to_app(z3_ctx, z3_literal);
    Z3_func_decl d = Z3_get_app_decl(z3_ctx, app);
    literal = Z3_func_decl_to_string(z3_ctx, d);

    found=literal.find("true");

    //std::cout << "literal2: " << literal << "\n";

    if (found!=std::string::npos)
      result=tvt(true);
    else
    {
      found=literal.find("false");
      if (found!=std::string::npos)
      {
        result=tvt(false);
      }
//      else
//    	return tvt(tvt::TV_ASSUME);
#if 1
      else
      {
        found=literal.find("not");
        if (found!=std::string::npos)
        {
          //result=tvt(true);
          //result=tvt(false);
          return tvt(tvt::TV_ASSUME);
        }
        else
        {
          found=literal.find("or");
		  found2=literal.find("bool");
          if (found!=std::string::npos && found2!=std::string::npos)
          {
            result=tvt(false);
          }
		  else if (found!=std::string::npos)
		  {
			result=tvt(true);
		  }
          else
          {
            found=literal.find("<=");
            if (found!=std::string::npos)
            {
              result=tvt(false);
            }
            else
            {
              found=literal.find("bvule");
              if (found!=std::string::npos)
              {
                result=tvt(true);
              }
              else
              {
                found=literal.find("=");
                found2=literal.find("int");
                if (found!=std::string::npos && found==std::string::npos)
                {
                  result=tvt(true);
                }
                else if (found!=std::string::npos && found2!=std::string::npos)
                {
                  result=tvt(false);
                }
                else
                {
                  found=literal.find("bvsle");
                  if (found!=std::string::npos)
                  {
                    //result=tvt(true);
                	return tvt(tvt::TV_ASSUME);
                  }
                }
              }
            }
          }
        }
      }
#endif
    }
  }

  if (a.sign())
	result=!result;

  return result;
}

#if 0
void z3_propt::set_assignment(literalt literal, bool value)
{
  std::cout << "value: " << value << std::endl;
}
#endif

void
z3_propt::assert_formula(Z3_ast ast, bool needs_literal)
{

  // If we're not going to be using the assumptions (ie, for unwidening and for
  // smtlib) then just assert the fact to be true.
  if (!store_assumptions) {
    Z3_assert_cnstr(z3_ctx, ast);
    return;
  }

  if (!needs_literal) {
    Z3_assert_cnstr(z3_ctx, ast);
    assumpt.push_front(ast);
  } else {
    literalt l = new_variable();
    Z3_ast formula = Z3_mk_iff(z3_ctx, z3_literal(l), ast);
    Z3_assert_cnstr(z3_ctx, formula);

    if (smtlib)
      assumpt.push_front(ast);
    else
      assumpt.push_front(z3_literal(l));
  }

  return;
}

void
z3_propt::assert_literal(literalt l, Z3_ast formula)
{

  Z3_assert_cnstr(z3_ctx, formula);
  if (store_assumptions) {
    if (smtlib)
      assumpt.push_front(formula);
    else
      assumpt.push_front(z3_literal(l));
  }

  return;
}
