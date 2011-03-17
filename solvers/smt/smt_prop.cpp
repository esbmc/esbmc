/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

Revisions: Roberto Bruttomesso, roberto.bruttomesso@unisi.ch

\*******************************************************************/

#include <assert.h>

#include <set>

#include <i2string.h>

#include "smt_prop.h"

/*******************************************************************\

Function: smt_propt::smt_propt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

smt_propt::smt_propt(std::ostream &_out):out(_out)
{
  // we skip index 0
  //
  // RB: Why ? In cvc_propt::cvc_propt it is set to 0
  // well, I'll leave it to 1 for now
  //
  _no_variables=1;
}

/*******************************************************************\

Function: smt_propt::land

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_propt::land(literalt a, literalt b, literalt o)
{
  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; lcnf" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << ":assumption" << std::endl;
  out << " (iff (and " << smt_literal(a) << " " << smt_literal(b) << ") " 
      << smt_literal(o) << ")" << std::endl;
}

/*******************************************************************\

Function: smt_propt::lor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_propt::lor(literalt a, literalt b, literalt o)
{
  out << ":assumption" << std::endl;
  out << " (iff (or " << smt_literal(a) << " " << smt_literal(b) << ") " 
      << smt_literal(o) << ")" << std::endl;
}

/*******************************************************************\

Function: smt_propt::lxor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_propt::lxor(literalt a, literalt b, literalt o)
{
  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; lxor" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << ":assumption" << std::endl;
  out << " (iff (xor " << smt_literal(a) << " " << smt_literal(b) << ") " 
      << smt_literal(o) << ")" << std::endl;
}

/*******************************************************************\

Function: smt_propt::lnand

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_propt::lnand(literalt a, literalt b, literalt o)
{
  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; lnand" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << ":assumption" << std::endl;
  out << " (iff (not (and " << smt_literal(a) << " " << smt_literal(b) << ")) " 
      << smt_literal(o) << ")" << std::endl;
}

/*******************************************************************\

Function: smt_propt::lnor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_propt::lnor(literalt a, literalt b, literalt o)
{
  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; lnor" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << ":assumption" << std::endl;
  out << " (iff (not (or " << smt_literal(a) << " " << smt_literal(b) << ")) " 
      << smt_literal(o) << ")" << std::endl;
}

/*******************************************************************\

Function: smt_propt::lequal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_propt::lequal(literalt a, literalt b, literalt o)
{
  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; lequal" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << ":assumption" << std::endl;
  out << " (iff (iff " << smt_literal(a) << " " << smt_literal(b) << ") " 
      << smt_literal(o) << ")" << std::endl;
}
  
/*******************************************************************\

Function: smt_propt::limplies

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_propt::limplies(literalt a, literalt b, literalt o)
{
  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; limplies" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << ":assumption" << std::endl;
  out << " (iff (implies " << smt_literal(a) << " " << smt_literal(b) << ") " 
      << smt_literal(o) << ")" << std::endl;
}

/*******************************************************************\

Function: smt_propt::land

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::land(const bvt &bv)
{
  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; land" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "(and";

  literalt literal=def_smt_literal();

  for(unsigned int i=0; i<bv.size(); ++i)
    out << " " << smt_literal(bv[i]);
  
  out << ")" << std::endl;

  return literal;  
}
  
/*******************************************************************\

Function: smt_propt::lor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::lor(const bvt &bv)
{
  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; lor" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;

  literalt literal=def_smt_literal();

  out << "(or";

  for(unsigned int i=0; i<bv.size(); ++i)
    out << " " << smt_literal(bv[i]);
  
  out << ")" << std::endl;

  return literal;  
}
  
/*******************************************************************\

Function: smt_propt::lxor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::lxor(const bvt &bv)
{
  if(bv.size()==0) return const_literal(false);
  if(bv.size()==1) return bv[0];
  if(bv.size()==2) return lxor(bv[0], bv[1]);

  literalt literal=const_literal(false);

  for(unsigned i=0; i<bv.size(); i++)
    literal=lxor(bv[i], literal);

  return literal;
}
  
/*******************************************************************\

Function: smt_propt::land

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::land(literalt a, literalt b)
{
  if(a==const_literal(true)) return b;
  if(b==const_literal(true)) return a;
  if(a==const_literal(false)) return const_literal(false);
  if(b==const_literal(false)) return const_literal(false);
  if(a==b) return a;

  literalt o=def_smt_literal();
  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; land" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  
  out << "(and " << smt_literal(a) << " " << smt_literal(b) << ")" << std::endl;

  return o;
}

/*******************************************************************\

Function: smt_propt::lor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::lor(literalt a, literalt b)
{
  if(a==const_literal(false)) return b;
  if(b==const_literal(false)) return a;
  if(a==const_literal(true)) return const_literal(true);
  if(b==const_literal(true)) return const_literal(true);
  if(a==b) return a;
  
  literalt o=def_smt_literal();
  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; lor" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "(or " << smt_literal(a) << " " << smt_literal(b) << ")" << std::endl;

  return o;
}

/*******************************************************************\

Function: smt_propt::lnot

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::lnot(literalt a)
{
  a.invert();
  return a;
}

/*******************************************************************\

Function: smt_propt::lxor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::lxor(literalt a, literalt b)
{
  if(a==const_literal(false)) return b;
  if(b==const_literal(false)) return a;
  if(a==const_literal(true)) return lnot(b);
  if(b==const_literal(true)) return lnot(a);

  literalt o=def_smt_literal();
  
  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; lxor" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "(xor " << smt_literal(a) << " " << smt_literal(b) << ")" << std::endl;

  return o;
}

/*******************************************************************\

Function: smt_propt::lnand

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::lnand(literalt a, literalt b)
{
  return lnot(land(a, b));
}

/*******************************************************************\

Function: smt_propt::lnor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::lnor(literalt a, literalt b)
{
  return lnot(lor(a, b));
}

/*******************************************************************\

Function: smt_propt::lequal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::lequal(literalt a, literalt b)
{
  return lnot(lxor(a, b));
}

/*******************************************************************\

Function: smt_propt::limplies

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::limplies(literalt a, literalt b)
{
  return lor(lnot(a), b);
}

/*******************************************************************\

Function: smt_propt::lselect

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::lselect(literalt a, literalt b, literalt c)
{ 
  if(a==const_literal(true)) return b;
  if(a==const_literal(false)) return c;
  if(b==c) return b;

  assert( false && "construct not supported yet" );

  out << std::endl;
  out << ";----------------------------------------------------------" << std::endl;
  out << "; lselect" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;

  literalt o=def_smt_literal();

  out << "IF " << smt_literal(a) << " THEN "
      << smt_literal(b) << " ELSE "
      << smt_literal(c) << " ENDIF;"
      << std::endl << std::endl;

  return o;
}

/*******************************************************************\

Function: smt_propt::new_variable

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::new_variable()
{
  out << ":extrapreds(( l" << _no_variables << " ))" << std::endl;
  literalt l;
  l.set(_no_variables, false);
  _no_variables++;
  return l;
}

/*******************************************************************\

Function: smt_propt::def_smt_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_propt::def_smt_literal()
{
  assert( false && "construct not supported yet" );

  out << "l" << _no_variables << ": BOOLEAN = ";
  literalt l;
  l.set(_no_variables, false);
  _no_variables++;
  return l;
}

/*******************************************************************\

Function: smt_propt::lcnf

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_propt::lcnf(const bvt &bv)
{
  if(bv.empty()) return;
  bvt new_bv;

  std::set<literalt> s;

  new_bv.reserve(bv.size());

  for(bvt::const_iterator it=bv.begin(); it!=bv.end(); it++)
  {
    if(s.insert(*it).second)
      new_bv.push_back(*it);

    if(s.find(lnot(*it))!=s.end())
      return; // clause satisfied

    // RB: Added first condition, I hope it makes sense
    assert( it->var_no( ) == literalt::const_var_no( )
	 || it->var_no( ) <= _no_variables);
  }

  assert(!new_bv.empty());

  out << ";----------------------------------------------------------" << std::endl;
  out << "; lcnf" << std::endl;
  out << ";----------------------------------------------------------" << std::endl;

  out << ":assumption" << std::endl;
  out << " ";

  if(new_bv.size()==1)
  {
    out << smt_literal(new_bv.front()) << std::endl;
  }
  else
  {
    out << "(and ";

    for(bvt::const_iterator it=new_bv.begin(); it!=new_bv.end(); it++)
      out << smt_literal(*it) << " ";

    out << ")" << std::endl;
  }

  out << std::endl;
}

/*******************************************************************\

Function: smt_propt::smt_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string smt_propt::smt_literal(literalt l)
{
  if(l==const_literal(false))
    return "false";
  else if(l==const_literal(true))
    return "true";

  if(l.sign())
    return "(not l"+i2string(l.var_no())+")";  

  return "l"+i2string(l.var_no());
}

/*******************************************************************\

Function: smt_propt::l_get

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

tvt smt_propt::l_get(literalt literal) const
{
  unsigned v=literal.var_no();
  if(v>=assignment.size()) return tvt(tvt::TV_UNKNOWN);
  tvt r=assignment[v];
  return literal.sign()?!r:r;
}

/*******************************************************************\

Function: smt_propt::prop_solve

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

propt::resultt smt_propt::prop_solve()
{
  return P_ERROR;
}
