/*******************************************************************\

Module: GOTO Programs

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <location.h>
#include <i2string.h>
#include <expr_util.h>
#include <guard.h>
#include <simplify_expr.h>
#include <array_name.h>
#include <arith_tools.h>
#include <base_type.h>

#include "goto_check.h"

class goto_checkt
{
public:
  goto_checkt(
    const namespacet &_ns,
    optionst &_options):
    ns(_ns),
    options(_options) { use_boolector=true; }

  void goto_check(goto_programt &goto_program);

protected:
  const namespacet &ns;
  optionst &options;
  bool use_boolector;

  void check_rec(const exprt &expr, guardt &guard, bool address);
  void check(const exprt &expr);

  void bounds_check(const exprt &expr, const guardt &guard);
  void div_by_zero_check(const exprt &expr, const guardt &guard);
  void pointer_rel_check(const exprt &expr, const guardt &guard);
  //void array_size_check(const exprt &expr);
  void overflow_check(const exprt &expr, const guardt &guard);
  void nan_check(const exprt &expr, const guardt &guard);
  std::string array_name(const exprt &expr);

  void add_guarded_claim(
    const exprt &expr,
    const std::string &comment,
    const std::string &property,
    const locationt &location,
    const guardt &guard);

  goto_programt new_code;
  std::set<exprt> assertions;
};

/*******************************************************************\

Function: goto_checkt::div_by_zero_check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_checkt::div_by_zero_check(
  const exprt &expr,
  const guardt &guard)
{
  if(options.get_bool_option("no-div-by-zero-check"))
    return;

  if(expr.operands().size()!=2)
    throw expr.id_string()+" takes two arguments";

  // add divison by zero subgoal

  exprt zero=gen_zero(expr.op1().type());

  if(zero.is_nil())
    throw "no zero of argument type of operator "+expr.id_string();

  exprt inequality("notequal", bool_typet());
  inequality.copy_to_operands(expr.op1(), zero);

  add_guarded_claim(
    inequality,
    "division by zero",
    "division-by-zero",
    expr.find_location(),
    guard);
}

/*******************************************************************\

Function: goto_checkt::overflow_check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_checkt::overflow_check(
  const exprt &expr,
  const guardt &guard)
{
  if(!options.get_bool_option("overflow-check"))
    return;

  // first, check type
  if(expr.type().id()!="signedbv")
    return;

  // add overflow subgoal

  exprt overflow("overflow-"+expr.id_string(), bool_typet());
  overflow.operands()=expr.operands();

  if(expr.id()=="typecast")
  {
    if(expr.operands().size()!=1)
      throw "typecast takes one operand";

    const typet &old_type=expr.op0().type();

    unsigned new_width=atoi(expr.type().width().c_str());
    unsigned old_width=atoi(old_type.width().c_str());

    if(old_type.id()=="unsignedbv") new_width--;
    if(new_width>=old_width) return;

    overflow.id(overflow.id_string()+"-"+i2string(new_width));
  }

  overflow.make_not();

  add_guarded_claim(
    overflow,
    "arithmetic overflow on "+expr.id_string(),
    "overflow",
    expr.find_location(),
    guard);

//  if (expr.id()=="*")
//    options.set_option("no-assume-guarantee", true);

  //options.set_option("int-encoding", false);
  if (!options.get_bool_option("eager"))
    options.set_option("no-assume-guarantee", false);
}

/*******************************************************************\

Function: goto_checkt::nan_check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_checkt::nan_check(
  const exprt &expr,
  const guardt &guard)
{
  if(!options.get_bool_option("nan-check"))
    return;

  // first, check type
  if(expr.type().id()!="floatbv")
    return;

  if(expr.id()!="+" &&
     expr.id()!="*" &&
     expr.id()!="/" &&
     expr.id()!="-")
    return;

  // add nan subgoal

  exprt isnan("isnan", bool_typet());
  isnan.copy_to_operands(expr);

  isnan.make_not();

  add_guarded_claim(
    isnan,
    "NaN on "+expr.id_string(),
    "NaN",
    expr.find_location(),
    guard);
}

/*******************************************************************\

Function: goto_checkt::pointer_rel_check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_checkt::pointer_rel_check(
  const exprt &expr,
  const guardt &guard)
{
  if(expr.operands().size()!=2)
    throw expr.id_string()+" takes one argument";

  if(expr.op0().type().id()=="pointer" &&
     expr.op1().type().id()=="pointer")
  {
    // add same-object subgoal

    if(!options.get_bool_option("no-pointer-check"))
    {
      exprt same_object("same-object", bool_typet());
      same_object.copy_to_operands(expr.op0(), expr.op1());
      //std::cout << "same_object.pretty(): " << same_object.pretty() << std::endl;
      add_guarded_claim(
        same_object,
        "same object violation",
        "pointer",
        expr.find_location(),
        guard);
    }
  }
}

/*******************************************************************\

Function: goto_checkt::array_name

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string goto_checkt::array_name(const exprt &expr)
{
  return ::array_name(ns, expr);
}

/*******************************************************************\

Function: goto_checkt::bounds_check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_checkt::bounds_check(
  const exprt &expr,
  const guardt &guard)
{
  if(options.get_bool_option("no-bounds-check"))
    return;

  if(expr.id()!="index")
    return;

  if(expr.is_bounds_check_set() && !expr.bounds_check())
    return;

  if(expr.operands().size()!=2)
    throw "index takes two operands";

  typet array_type=ns.follow(expr.op0().type());

  if(array_type.id()=="pointer")
    return; // done by the pointer code
  else if(array_type.id()=="incomplete_array")
  {
    std::cerr << expr.pretty() << std::endl;
    throw "index got incomplete array";
  }
  else if(!array_type.is_array())
    throw "bounds check expected array type, got "+array_type.id_string();

  std::string name=array_name(expr.op0());

  const exprt &index=expr.op1();

  if(index.type().id()!="unsignedbv")
  {
    // we undo typecasts to signedbv
    if(index.id()=="typecast" &&
       index.operands().size()==1 &&
       index.op0().type().id()=="unsignedbv")
    {
      // ok
    }
    else
    {
      mp_integer i;

      if(!to_integer(index, i) && i>=0)
      {
        // ok
      }
      else
      {
        exprt zero=gen_zero(index.type());

        if(zero.is_nil())
          throw "no zero constant of index type "+
            index.type().to_string();

        exprt inequality(">=", bool_typet());
        inequality.copy_to_operands(index, zero);

        add_guarded_claim(
          inequality,
          name+" lower bound",
          "array bounds",
          expr.find_location(),
          guard);
      }
    }
  }

  {
    if(array_type.size_irep().is_nil())
      throw "index array operand of wrong type";

    const exprt &size=(const exprt &)array_type.size_irep();

    if(size.id()!="infinity")
    {
      exprt inequality("<", bool_typet());
      inequality.copy_to_operands(index, size);

      // typecast size
      if(inequality.op1().type()!=inequality.op0().type())
        inequality.op1().make_typecast(inequality.op0().type());

      //std::cout << "inequality.pretty(): " << inequality.pretty() << std::endl;
      //std::cout << "guard.as_expr().pretty(): " << guard.as_expr().pretty() << std::endl;
      add_guarded_claim(
        inequality,
        name+" upper bound",
        "array bounds",
        expr.find_location(),
        guard);
    }
  }
}

/*******************************************************************\

Function: goto_checkt::array_size_check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#if 0
void goto_checkt::array_size_check(
  const exprt &expr,
  const irept &location)
{
  if(expr.type().is_array())
  {
    const exprt &size=(exprt &)expr.type().size_irep();

    if(size.type().id()=="unsignedbv")
    {
      // nothing to do
    }
    else
    {
      exprt zero=gen_zero(size.type());

      if(zero.is_nil())
        throw "no zero constant of index type "+
          size.type().to_string();

      exprt inequality(">=", bool_typet());
      inequality.copy_to_operands(size, zero);

      std::string name=array_name(expr);

      guardt guard;
      add_guarded_claim(inequality, name+" size", location, guard);
    }
  }
}
#endif

/*******************************************************************\

Function: goto_checkt::add_guarded_claim

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_checkt::add_guarded_claim(
  const exprt &_expr,
  const std::string &comment,
  const std::string &property,
  const locationt &location,
  const guardt &guard)
{
  bool all_claims=options.get_bool_option("all-claims");
  exprt expr(_expr);

  // first try simplifier on it
  if(!options.get_bool_option("no-simplify"))
  {
    base_type(expr, ns);
    simplify(expr);
  }

  if(!all_claims && expr.is_true())
    return;

  // add the guard
  exprt guard_expr=guard.as_expr();

  exprt new_expr;

  if(guard_expr.is_true())
    new_expr.swap(expr);
  else
  {
    new_expr=exprt("=>", bool_typet());
    new_expr.move_to_operands(guard_expr, expr);
  }

  if(assertions.insert(new_expr).second)
  {
    goto_programt::targett t=new_code.add_instruction(ASSERT);

    t->guard.swap(new_expr);
    t->location=location;
    t->location.comment(comment);
    t->location.property(property);
  }
}

/*******************************************************************\

Function: goto_checkt::check_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_checkt::check_rec(
  const exprt &expr,
  guardt &guard,
  bool address)
{
  if(address)
  {
    if(expr.is_dereference())
    {
      assert(expr.operands().size()==1);
      check_rec(expr.op0(), guard, false);
    }
    else if(expr.id()=="index")
    {
      assert(expr.operands().size()==2);
      check_rec(expr.op0(), guard, true);
      check_rec(expr.op1(), guard, false);
    }
    else
    {
      forall_operands(it, expr)
        check_rec(*it, guard, true);
    }
    return;
  }

  if(expr.is_address_of())
  {
    assert(expr.operands().size()==1);
    check_rec(expr.op0(), guard, true);
    return;
  }
  else if(expr.is_and() || expr.id()=="or")
  {
    if(!expr.is_boolean())
      throw expr.id_string()+" must be Boolean, but got "+
            expr.pretty();

    unsigned old_guards=guard.size();

    for(unsigned i=0; i<expr.operands().size(); i++)
    {
      const exprt &op=expr.operands()[i];

      if(!op.is_boolean())
        throw expr.id_string()+" takes Boolean operands only, but got "+
              op.pretty();

      check_rec(op, guard, false);

      if(expr.id()=="or")
      {
        exprt tmp(op);
        tmp.make_not();
        guard.move(tmp);
      }
      else
        guard.add(op);
    }

    guard.resize(old_guards);

    return;
  }
  else if(expr.id()=="if")
  {
    if(expr.operands().size()!=3)
      throw "if takes three arguments";

    if(!expr.op0().is_boolean())
    {
      std::string msg=
        "first argument of if must be boolean, but got "
        +expr.op0().to_string();
      throw msg;
    }

    check_rec(expr.op0(), guard, false);

    {
      unsigned old_guard=guard.size();
      guard.add(expr.op0());
      check_rec(expr.op1(), guard, false);
      guard.resize(old_guard);
    }

    {
      unsigned old_guard=guard.size();
      exprt tmp(expr.op0());
      tmp.make_not();
      guard.move(tmp);
      check_rec(expr.op2(), guard, false);
      guard.resize(old_guard);
    }

    return;
  }

  forall_operands(it, expr)
    check_rec(*it, guard, false);

  //std::cout << "expr.id(): " << expr.id() << std::endl;
  if(expr.id()=="index")
  {
    bounds_check(expr, guard);
  }
  else if(expr.id()=="/")
  {
    div_by_zero_check(expr, guard);
    nan_check(expr, guard);
  }
  else if(expr.id()=="+" || expr.id()=="-" ||
          expr.id()=="*" ||
          expr.id()=="unary-" ||
          expr.id()=="typecast")
  {
    if(expr.type().id()=="signedbv")
    {
      overflow_check(expr, guard);
      if (expr.id()=="typecast" && expr.op0().type().id()!="signedbv")
      {
   		if (!options.get_bool_option("boolector-bv") && !options.get_bool_option("z3-bv")
   			&& !options.get_bool_option("z3-ir"))
   		{
   		  options.set_option("int-encoding", false);
   		}
      }
    }
    else if(expr.type().id()=="floatbv")
      nan_check(expr, guard);
  }
  else if(expr.id()=="<=" || expr.id()=="<" ||
          expr.id()==">=" || expr.id()==">")
  {
    pointer_rel_check(expr, guard);

  }

  if (expr.id() == "ashr" || expr.id() == "lshr" ||
	  expr.id() == "shl")
  {
    if (!options.get_bool_option("boolector-bv") && !options.get_bool_option("z3-bv")
		&& !options.get_bool_option("z3-ir"))
    {
      if (use_boolector)
        options.set_option("bl", true);
      else
        options.set_option("int-encoding", false);
    }
  }
  else if (expr.id() == "bitand" || expr.id() == "bitor" ||
		   expr.id() == "bitxor" || expr.id() == "bitnand" ||
		   expr.id() == "bitnor" || expr.id() == "bitnxor")
  {
	if (!options.get_bool_option("boolector-bv") && !options.get_bool_option("z3-bv")
		&& !options.get_bool_option("z3-ir"))
	{
	  if (use_boolector)
        options.set_option("bl", true);
    }
  }
  else if (expr.id() == "mod")
  {
    div_by_zero_check(expr, guard);
    nan_check(expr, guard);

	if (!options.get_bool_option("boolector-bv") && !options.get_bool_option("z3-bv")
		&& !options.get_bool_option("z3-ir"))
	{
	  if (use_boolector)
	    options.set_option("bl", true);
	  else
        options.set_option("int-encoding", false);
	}
  }
  else if (expr.id() == "struct" || expr.id() == "union"
		    || expr.type().id()=="pointer" || expr.id()=="member" ||
		    (expr.type().is_array() && expr.type().subtype().is_array()))
  {
	use_boolector=false; //always deactivate boolector
	options.set_option("bl", false);
	options.set_option("z3", true); //activate Z3 for solving the VCs
  }
  else if (expr.type().is_fixedbv())
  {
	use_boolector=false;
	options.set_option("bl", false);
	options.set_option("z3", true);
	if (!options.get_bool_option("z3-bv"))
      options.set_option("int-encoding", true);

	if (!options.get_bool_option("eager"))
	  options.set_option("no-assume-guarantee", false);
  }

  if (options.get_bool_option("qf_aufbv"))
  {
	use_boolector=false; //always deactivate boolector
	options.set_option("bl", false);
    options.set_option("z3", true); //activate Z3 to generate the file in SMT lib format
    options.set_option("int-encoding", false);
  }

  if (options.get_bool_option("qf_auflira"))
  {
	use_boolector=false; //always deactivate boolector
	options.set_option("bl", false);
    options.set_option("z3", true); //activate Z3 to generate the file in SMT lib format
    options.set_option("int-encoding", true);
  }

}

/*******************************************************************\

Function: goto_checkt::check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_checkt::check(const exprt &expr)
{
  guardt guard;
  check_rec(expr, guard, false);
}

/*******************************************************************\

Function: goto_checkt::goto_check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_checkt::goto_check(goto_programt &goto_program)
{
  for(goto_programt::instructionst::iterator
      it=goto_program.instructions.begin();
      it!=goto_program.instructions.end();
      it++)
  {
    goto_programt::instructiont &i=*it;

    new_code.clear();
    assertions.clear();

    check(i.guard);

    if(i.is_other())
    {
      const irep_idt &statement=i.code.statement();

      if(statement=="expression")
      {
        check(i.code);
      }
      else if(statement=="printf")
      {
        forall_operands(it, i.code)
          check(*it);
      }
    }
    else if(i.is_assign())
    {
      if(i.code.operands().size()!=2)
        throw "assignment expects two operands";

      check(i.code.op0());
      check(i.code.op1());
    }
    else if(i.is_function_call())
    {
      forall_operands(it, i.code)
        check(*it);
    }
    else if(i.is_return())
    {
      if(i.code.operands().size()==1)
        check(i.code.op0());
    }

    for(goto_programt::instructionst::iterator
        i_it=new_code.instructions.begin();
        i_it!=new_code.instructions.end();
        i_it++)
    {
      i_it->local_variables=it->local_variables;
      if(i_it->location.is_nil()) i_it->location=it->location;
      if(i_it->function=="") i_it->function=it->function;
      if(i_it->function=="") i_it->function=it->function;
    }

    // insert new instructions -- make sure targets are not moved

    while(!new_code.instructions.empty())
    {
      goto_program.insert_swap(it, new_code.instructions.front());
      new_code.instructions.pop_front();
      it++;
    }
  }
}

/*******************************************************************\

Function: goto_check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_check(
  const namespacet &ns,
  optionst &options,
  goto_programt &goto_program)
{
  goto_checkt goto_check(ns, options);
  goto_check.goto_check(goto_program);
}

/*******************************************************************\

Function: goto_check

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_check(
  const namespacet &ns,
  optionst &options,
  goto_functionst &goto_functions)
{
  goto_checkt goto_check(ns, options);

  for(goto_functionst::function_mapt::iterator
      it=goto_functions.function_map.begin();
      it!=goto_functions.function_map.end();
      it++)
  {
    goto_check.goto_check(it->second.body);
  }

}
