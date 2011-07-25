/*******************************************************************\

Module: Program Transformation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <i2string.h>
#include <replace_expr.h>
#include <expr_util.h>
#include <location.h>
#include <std_expr.h>
#include <config.h>
#include <std_expr.h>

#include <ansi-c/c_types.h>

#include <pointer-analysis/value_set_analysis.h>

#include "remove_skip.h"
#include "goto_convert_functions.h"

/*******************************************************************\

Function: build_if_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static exprt build_if_expr(
  const exprt &ptr_expr,
  const value_sett::expr_sett &expr_set)
{
  exprt dest;
  dest.make_nil();

  exprt *p=&dest;

  for(value_sett::expr_sett::const_iterator
      it=expr_set.begin();
      it!=expr_set.end();
      it++)
  {
    if(it->id()=="object_descriptor")
    {
      const object_descriptor_exprt &o=to_object_descriptor_expr(*it);

      exprt address=exprt("address_of", ptr_expr.type());
      address.copy_to_operands(o.object());
      
      if(p->is_nil())
        *p=o.object();
      else
      {
        equality_exprt equality;
        equality.lhs()=ptr_expr;
        equality.rhs()=address;
      
        exprt if_expr("if", o.object().type());
        if_expr.operands().resize(3);
        if_expr.op0().swap(equality);
        if_expr.op1()=o.object();
        if_expr.op2().swap(*p);
        
        p->swap(if_expr);
      }
    }
  }
  
  return dest;
}

/*******************************************************************\

 Function: value_sett::build_or_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static exprt build_or_expr(
  const exprt &ptr_expr,
  const value_sett::expr_sett &expr_set)
{
  exprt expr_or;
  expr_or.make_nil();

  for(value_sett::expr_sett::const_iterator
      it=expr_set.begin();
      it!=expr_set.end();
      it++)
  {
    if(it->id()=="object_descriptor")
    {
      const object_descriptor_exprt &o=
        to_object_descriptor_expr(*it);

      if(o.object().type().id()!="code") // bad pointer
        continue;

      pointer_typet pointer_type;
      pointer_type.subtype()=o.object().type();

      exprt address;

      if(o.object().id()=="NULL-object")
      {
        address=constant_exprt(pointer_type);
        address.value("NULL");
      }
      else
      {
        address=exprt("address_of", pointer_type);
        address.copy_to_operands(o.object());
      }

      equality_exprt equality;
      equality.lhs()=ptr_expr;
      equality.rhs()=address;
      
      if(equality.rhs().type()!=equality.lhs().type())
        equality.lhs().make_typecast(equality.rhs().type());
      
      if(expr_or.is_nil())
        expr_or=equality;
      else
      {
        exprt tmp("or", typet("bool"));
        tmp.move_to_operands(expr_or, equality);
        expr_or.swap(tmp);
      }
    }
  }

  if(expr_or.is_nil())
    expr_or.make_false();

  return expr_or;
}

/*******************************************************************\

Function: goto_convert_functionst::remove_function_pointers

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_convert_functionst::remove_function_pointers()
{
  // first see if there is something to do
  if(!have_function_pointers())
    return false;
    
  messaget message(message_handler);
  message.status("Function Pointer Removal");

  // get value sets
  value_set_analysist value_set_analysis(ns);

  value_set_analysis(functions);
  
  bool did_something=false;

  // remove function pointers
  for(goto_functionst::function_mapt::iterator f_it=
      functions.function_map.begin();
      f_it!=functions.function_map.end();
      f_it++)
    if(remove_function_pointers(value_set_analysis, f_it->second.body))
      did_something=true;

  if(did_something)
    functions.compute_location_numbers();
      
  return did_something;
}

/*******************************************************************\

Function: goto_convert_functionst::have_function_pointers

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_convert_functionst::have_function_pointers()
{
  for(goto_functionst::function_mapt::iterator f_it=
      functions.function_map.begin();
      f_it!=functions.function_map.end();
      f_it++)
    if(have_function_pointers(f_it->second.body))
      return true;

  return false;
}

/*******************************************************************\

Function: goto_convert_functionst::have_function_pointers

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_convert_functionst::have_function_pointers(
  const goto_programt &goto_program)
{
  for(goto_programt::const_targett
      target=goto_program.instructions.begin();
      target!=goto_program.instructions.end();
      target++)
    if(target->is_function_call())
    {
      const code_function_callt &code=
        to_code_function_call(target->code);
    
      if(code.function().id()=="dereference" ||
         code.function().id()=="implicit_dereference")
        return true;
    }

  return false;
}

/*******************************************************************\

Function: goto_convert_functionst::remove_function_pointers

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_convert_functionst::remove_function_pointers(
  value_setst &value_sets,
  goto_programt &goto_program)
{
  bool did_something=false;

  for(goto_programt::targett target=goto_program.instructions.begin();
      target!=goto_program.instructions.end();
      target++)
    if(target->is_function_call())
    {
      const code_function_callt &code=
        to_code_function_call(target->code);
    
      if(code.function().id()=="dereference" ||
         code.function().id()=="implicit_dereference")
      {
        remove_function_pointer(value_sets, goto_program, target); 
        did_something=true;
      }
    }

  if(did_something)
  {
    remove_skip(goto_program);
    goto_program.update();
  }
    
  return did_something;
}

/*******************************************************************\

Function: goto_convert_functionst::remove_function_pointer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convert_functionst::remove_function_pointer(
  value_setst &value_sets,
  goto_programt &goto_program,
  goto_programt::targett target)
{
  const code_function_callt &code=
    to_code_function_call(target->code);

  const exprt &lhs=code.lhs();
  const exprt &function=code.function();

  assert(function.id()=="dereference");  
  assert(function.operands().size()==1);

  value_setst::valuest value_set;
  std::set<exprt> new_value_set;

  value_sets.get_values(target, function.op0(), value_set);

  // remove unknown values
  bool has_unknown=false;

  for(value_setst::valuest::const_iterator
      it=value_set.begin();
      it!=value_set.end();
      it++)
  {
    if(it->id()=="unknown")
      has_unknown=true;
    else
      new_value_set.insert(*it);
  }
  
  if(has_unknown)
  {
    // add all type-compatible functions
    // that have a body
    forall_symbols(it, ns.get_context().symbols)
    {
      if(it->second.is_type ||
         it->second.type!=function.type() ||
         it->second.value.is_nil())
         continue;

      object_descriptor_exprt o;
      o.object()=symbol_expr(it->second);
      o.type()=it->second.type;
      o.offset()=gen_zero(index_type());

      new_value_set.insert(o);
    }
  }

  goto_programt new_code;

  //exStbDemo
#if 1
  if(!options.get_bool_option("no-pointer-check"))
  {
    // make sure the pointer is correct
    exprt expr_or=build_or_expr(function.op0(), new_value_set);
    goto_programt::targett t=
      new_code.add_instruction(ASSERT);
    t->guard=expr_or;
    t->location=function.location();
    t->location.set("comment", "unexpected target function");
  }
#endif

  exprt new_function=build_if_expr(function.op0(), new_value_set);

  if(new_function.is_not_nil())
    do_function_call(lhs, new_function, code.arguments(), new_code);
  else
  {
    err_location(function.location());
    str << "no target candidate for function call `"
        << from_expr(ns, "", function) << "'";
    warning();
  }
  
  finish_gotos();
  
  // fix local variables
  Forall_goto_program_instructions(it, new_code)
    it->local_variables.insert(target->local_variables.begin(),
                               target->local_variables.end());
  
  goto_programt::targett next_target=target;
  next_target++;

  goto_program.instructions.splice(next_target, new_code.instructions);

  // we preserve the dereferencing
  code_expressiont code_expression;
  code_expression.location()=function.location();
  code_expression.copy_to_operands(function);
  target->code.swap(code_expression);
  target->type=OTHER;
}
