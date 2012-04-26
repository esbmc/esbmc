/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <map>

#include <irep2.h>
#include "symbol.h"
#include "prop_conv.h"

bool prop_convt::literal(const exprt &expr, literalt &dest) const
{
  assert(expr.type().is_bool());

  if(expr.id()=="symbol")
  {
    const irep_idt &identifier=expr.identifier();

    symbolst::const_iterator result=symbols.find(identifier);

    if(result==symbols.end()) return true;
    dest=result->second;
    return false;
  }

  throw "found no literal for expression";
}

literalt prop_convt::get_literal(const irep_idt &identifier)
{
  std::pair<symbolst::iterator, bool> result=
    symbols.insert(std::pair<irep_idt, literalt>(identifier, literalt()));

  if(!result.second)
    return result.first->second;

  literalt literal=prop.new_variable();

  // insert
  result.first->second=literal;

  return literal;
}

bool prop_convt::get_bool(const exprt &expr, tvt &value) const
{
  // trivial cases

  if(expr.is_true())
  {
    value=tvt(true);
    return false;
  }
  else if(expr.is_false())
  {
    value=tvt(false);
    return false;
  }
  else if(expr.id()=="symbol")
  {
    symbolst::const_iterator result=symbols.find(expr.identifier());
    if(result==symbols.end()) return true;

    value=prop.l_get(result->second);
    return false;
  }

  // sub-expressions

  if(expr.id()=="not")
  {
    if(expr.type().is_bool() &&
       expr.operands().size()==1)
    {
      if(get_bool(expr.op0(), value)) return true;
      value=!value;
      return false;
    }
  }
  else if(expr.is_and() || expr.id()=="or")
  {
    if(expr.type().is_bool() &&
       expr.operands().size()>=1)
    {
      value=tvt(expr.is_and());

      forall_operands(it, expr)
      {
        tvt tmp;
        if(get_bool(*it, tmp)) return true;

        if(expr.is_and())
        {
          if(tmp.is_false()) { value=tvt(false); return false; }

          value=value && tmp;
        }
        else // or
        {
          if(tmp.is_true()) { value=tvt(true); return false; }

          value=value || tmp;
        }
      }

      return false;
    }
  }

  // check cache

  cachet::const_iterator cache_result=cache.find(expr);
  if(cache_result==cache.end()) return true;

  value=prop.l_get(cache_result->second);
  return false;
}

literalt prop_convt::convert(const exprt &expr, bool do_cache)
{
  if(!do_cache ||
     expr.id()=="symbol" ||
     expr.id()=="constant")
    return convert_bool(expr);

  // check cache first

  std::pair<cachet::iterator, bool> result=
    cache.insert(std::pair<exprt, literalt>(expr, literalt()));

  if(!result.second)
    return result.first->second;

  literalt literal=convert_bool(expr);

  // insert into cache

  result.first->second=literal;

  return literal;
}

literalt prop_convt::convert_bool(const exprt &expr)
{
  if(!expr.type().is_bool())
  {
    std::string msg="prop_convt::convert_bool got "
                    "non-boolean expression:\n";
    msg+=expr.to_string();
    throw msg;
  }

  const exprt::operandst &op=expr.operands();

  if(expr.is_constant())
  {
    if(expr.is_true())
      return const_literal(true);
    else if(expr.is_false())
      return const_literal(false);
    else
      throw "unknown boolean constant: "+expr.to_string();
  }

  return convert_rest(expr);
}

literalt prop_convt::convert_rest(const exprt &expr)
{
  // fall through
  ignoring(expr);
  return prop.new_variable();
}

bool prop_convt::set_equality_to_true(const exprt &expr)
{
  if(!equality_propagation) return true;

  if(expr.operands().size()==2)
  {
    // optimization for constraint of the form
    // new_variable = value

    if(expr.op0().id()=="symbol")
    {
      const irep_idt &identifier=
        expr.op0().identifier();

      literalt tmp=convert(expr.op1());

      std::pair<symbolst::iterator, bool> result=
        symbols.insert(std::pair<irep_idt, literalt>(identifier, tmp));

      if(result.second)
        return false; // ok, inserted!

      // nah, already there
    }
  }

  return true;
}

void prop_convt::ignoring(const exprt &expr)
{
  // fall through

  std::string msg="warning: ignoring "+expr.pretty();

  print(2, msg);
}

propt::resultt prop_convt::dec_solve()
{

  print(7, "Solving with "+prop.solver_text());

  return prop.prop_solve();
}

exprt prop_convt::get(const exprt &expr) const
{
  exprt dest;

  dest.make_nil();

  tvt value;

  if(expr.type().is_bool() &&
     !get_bool(expr, value))
  {
    switch(value.get_value())
    {
     case tvt::TV_TRUE:  dest.make_true(); return dest;
     case tvt::TV_FALSE: dest.make_false(); return dest;
     case tvt::TV_UNKNOWN: dest.make_false(); return dest; // default
    }
  }

  return dest;
}

void prop_convt::convert_smt_type(const type2t &type, void *&arg) const
{
  std::cerr << "Unhandled SMT conversion for type ID " << type.type_id <<
               std::endl;
  abort();
}

void prop_convt::convert_smt_expr(const expr2t &expr, void *&arg)
{
  std::cerr << "Unhandled SMT conversion for expr ID " << expr.expr_id <<
               std::endl;
  abort();
}
