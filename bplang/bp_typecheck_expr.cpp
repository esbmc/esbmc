/*******************************************************************\

Module: Boolean Program Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include "bp_typecheck.h"

/*******************************************************************\

Function: bp_typecheckt::typecheck_boolean_operands

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bp_typecheckt::typecheck_boolean_operands(exprt &expr)
{
  if(expr.operands().size()==0)
  {
    err_location(expr);
    str << "Expected operands for " << expr.id_string()
        << " operator";
    throw 0;
  }

  Forall_operands(it, expr)
    typecheck_expr(*it);

  expr.type()=typet("bool");

  forall_operands(it, expr)
    if(it->type().id()!="bool")
    {
      err_location(*it);
      str << "Expected Boolean operands for " << expr.id_string()
          << " operator";
      throw 0;
    }
}

/*******************************************************************\

Function: bp_typecheckt::typecheck_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bp_typecheckt::typecheck_expr(exprt &expr)
{
  if(expr.id()=="symbol")
  {
    const irep_idt &identifier=expr.get("identifier");

    if(identifier=="T")
      expr.make_true();
    else if(identifier=="F")
      expr.make_false();
    else if(identifier=="_")
    {
      expr.id("bp_unused");
      expr.remove("identifier");
    }
    else
    {
      irep_idt full_identifier;
      
      // first try local variable
      full_identifier=
        "bp::local_var::"+
        id2string(function_name)+"::"+
        id2string(identifier);

      symbolst::iterator s_it=context.symbols.find(full_identifier);

      if(s_it==context.symbols.end())
      {
        // try function argument next
        full_identifier=
          "bp::argument::"+id2string(function_name)+"::"+
          id2string(identifier);

        s_it=context.symbols.find(full_identifier);

        if(s_it==context.symbols.end())
        {
          // try global next
          full_identifier="bp::var::"+id2string(identifier);

          s_it=context.symbols.find(full_identifier);

          if(s_it==context.symbols.end())
          {
            err_location(expr);
            str << "variable `" << identifier << "' not found";
            throw 0;
          }
        }
      }

      symbolt &symbol=s_it->second;

      expr.type()=symbol.type;
      expr.set("identifier", full_identifier);
    }
  }
  else if(expr.id()=="nondet_bool")
  {
    expr.type()=typet("bool");
  }
  else if(expr.id()=="tick")
  {
    if(expr.operands().size()!=1)
    {
      err_location(expr);
      throw "tick operator expects one operand";
    }

    exprt &op=expr.op0();
    
    typecheck_expr(op);

    if(op.id()!="symbol")
    {
      err_location(expr);
      throw "tick operator expects a symbol as operand";
    }
    
    exprt tmp;
    
    tmp.swap(op);
    tmp.id("next_symbol");
    
    expr.swap(tmp);
  }
  else if(expr.id()=="bp_schoose")
  {
    typecheck_boolean_operands(expr);
    
    // Must have two arguments.
    // The first one is for "result must be true",
    // the second for "result must be false"
    // schoose[a,b] is like (* and !b) | a
    
    if(expr.operands().size()!=2)
    {
      err_location(expr);
      throw "schoose takes two arguments";
    }
    
    if(expr.op0().is_true())
    {
      // schoose[T,F] means "must be true and not false"
      // schoose[T,T] means "must be true and false"
      // in Moped, the later happens to be true
      expr.make_true();
    }
    else if(expr.op1().is_true())
    {
      // schoose[x,T] is x
      exprt tmp;
      tmp.swap(expr.op0());
      expr.swap(tmp);
    }
    else if(expr.op0().is_false() &&
            expr.op1().is_false())
    {
      // schoose[F,F] is true non-deterministic choice
      expr.id("nondet_bool");
      expr.operands().clear();
    }
    else if(expr.op1().is_false())
    {
      // schoose[x,T] is * | x

      exprt nondet("nondet_bool", typet("bool"));
      exprt or_expr("or", typet("bool"));
      or_expr.move_to_operands(nondet, expr.op0());
      expr.swap(or_expr);
    }
    else if(expr.op1().id()=="not" &&
            expr.op1().operands().size()==1 &&
            expr.op1().op0()==expr.op0())
    {
      // schoose[a,!a]==a
      exprt tmp;
      tmp.swap(expr.op0());
      expr.swap(tmp);
    }
    else // Variables involved...
    {
      // build (* and !b) | a
      exprt nondet("nondet_bool", typet("bool"));
      exprt not_op1("not", typet("bool"));
      not_op1.move_to_operands(expr.op1());
      exprt and_expr("and", typet("bool"));
      and_expr.move_to_operands(nondet, not_op1);
      exprt or_expr("or", typet("bool"));
      or_expr.move_to_operands(and_expr, expr.op0());
      expr.swap(or_expr);
    }
  }
  else if(expr.id()=="and" || expr.id()=="or" ||
          expr.id()=="xor" || expr.id()=="not" ||
          expr.id()=="=>" ||
          expr.id()=="=" || expr.id()=="notequal")
  {
    typecheck_boolean_operands(expr);
  }
  else if(expr.id()=="constant")
  {
    const irep_idt &value=expr.get("value");
    
    if(value=="0")
    {
      expr.make_false();
    }
    else if(value=="1")
    {
      expr.make_true();
    }
    else
    {
      err_location(expr);
      str << "Invalid constant: " << value;
      throw 0;
    }
  }
  else if(expr.id()=="sideeffect")
  {
    //const irep_idt &statement=expr.get("statement");

    err_location(expr);
    str << "No type checking for sideeffect " << expr;
    throw 0;
  }
  else
  {
    err_location(expr);
    str << "No type checking for " << expr;
    throw 0;
  }
}
