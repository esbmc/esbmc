/*******************************************************************\

Module: Program Transformation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <i2string.h>
#include <replace_expr.h>
#include <expr_util.h>
#include <location.h>
#include <cprover_prefix.h>
#include <prefix.h>
#include <arith_tools.h>
#include <simplify_expr.h>
#include <std_code.h>
#include <std_expr.h>

#include <ansi-c/c_types.h>

#include "goto_convert_class.h"

#define TSE_PAPER 1

/*******************************************************************\

Function: get_alloc_type_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void get_alloc_type_rec(
  const exprt &src,
  typet &type,
  exprt &size)
{
  static bool is_mul=false;

  const irept &sizeof_type=src.c_sizeof_type();
  //nec: ex33.c
  if(!sizeof_type.is_nil() && !is_mul)
  {
    type=(typet &)sizeof_type;
  }
  else if(src.id()=="*")
  {
	is_mul=true;
    forall_operands(it, src)
      get_alloc_type_rec(*it, type, size);
  }
  else
  {
    size.copy_to_operands(src);
  }
}

/*******************************************************************\

Function: get_alloc_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void get_alloc_type(
  const exprt &src,
  typet &type,
  exprt &size)
{
  type.make_nil();
  size.make_nil();

  get_alloc_type_rec(src, type, size);

  if(type.is_nil())
    type=char_type();

  if(size.has_operands())
  {
    if(size.operands().size()==1)
    {
      exprt tmp;
      tmp.swap(size.op0());
      size.swap(tmp);
    }
    else
    {
      size.id("*");
      size.type()=size.op0().type();
    }
  }
}

/*******************************************************************\

Function: goto_convertt::do_pthread_create

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
#if 1
void goto_convertt::do_pthread_create(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(arguments.size()!=4)
  {
    err_location(lhs);
    throw "phread_create takes four arguments";
  }

  // arguments:
  // pthread_t *__threadp,
  //  const pthread_attr_t *__attr,
  // void *(*__start_routine) (void *),
  // void *__arg

  // first do non-det assignment to __threadp

  {
    if(arguments[0].type().id()!="pointer")
    {
      err_location(arguments[0]);
      throw "first argument of pthread_create is expected to be"
            " of a pointer type";
    }

    if(arguments[0].type().subtype().id()!="empty")
    {
      exprt lhs("dereference", arguments[0].type().subtype());
      lhs.copy_to_operands(arguments[0]);
      lhs.location()=arguments[0].location();

      exprt rhs = side_effect_expr_nondett(lhs.type());
      rhs.location()=function.location();

      code_assignt assignment(lhs, rhs);
      assignment.location()=function.location();

      convert(assignment, dest);
    }
  }

  // do the function call
  {
    code_function_callt function_call;

    exprt &thread_function=function_call.function();

    thread_function=arguments[2];
    function_call.arguments().push_back(arguments[3]);
    function_call.location()=function.location();

    if(thread_function.type().id()!="pointer")
    {
      err_location(function);
      throw "create_thread expects function pointer as third argument";
    }

    if(!thread_function.type().subtype().is_code())
    {
      // cast it to code type
      code_typet ct;
      ct.return_type()=empty_typet();
      ct.arguments().resize(1);
      ct.arguments()[0].type()=pointer_typet();
      ct.arguments()[0].type().subtype()=empty_typet();

      pointer_typet pt;
      pt.subtype()=ct;

      thread_function.make_typecast(pt);
    }

    // do dereferencing
    if(thread_function.id()=="implicit_address_of" ||
       thread_function.is_address_of())
    {
      exprt tmp;
      assert(thread_function.operands().size()==1);
      tmp.swap(thread_function.op0());
      thread_function.swap(tmp);
    }
    else
    {
      exprt tmp("dereference", thread_function.type().subtype());
      tmp.move_to_operands(thread_function);
      tmp.location()=function.location();
      thread_function.swap(tmp);
    }

    codet start_thread("start_thread");
    start_thread.move_to_operands(function_call);

  	/////////////////////
	if(arguments[0].type().subtype().id()!="empty")
	{
 		exprt lhs("dereference", arguments[0].type().subtype());
 		lhs.copy_to_operands(arguments[0]);
      	lhs.location()=arguments[0].location();
		start_thread.move_to_operands(lhs);
	}
    /////////////////////

    start_thread.location()=function.location();
    convert(start_thread, dest);
  }

  if(lhs.is_not_nil())
  {
    // return 0, boring
    exprt zero=gen_zero(lhs.type());
    if(zero.is_not_nil())
    {
      code_assignt assignment(lhs, zero);
      assignment.location()=function.location();
      //convert(assignment, dest);
    }
  }

}
#endif
/*******************************************************************\

Function: goto_convertt::do_pthread_exit

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_pthread_exit(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(arguments.size()!=1)
  {
    err_location(lhs);
    throw "phread_exit takes one argument";
  }

  goto_programt::targett t=dest.add_instruction(END_THREAD);

  t->location=function.location();
}

/*******************************************************************\

Function: goto_convertt::do_printf

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_printf(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  const irep_idt &f_id=function.identifier();

  if(f_id==CPROVER_PREFIX "printf" ||
     f_id=="c::printf")
  {
    exprt printf_code("sideeffect",
      static_cast<const typet &>(function.type().return_type()));

    printf_code.statement("printf");

    printf_code.operands()=arguments;
    printf_code.location()=function.location();

    if(lhs.is_not_nil())
    {
      code_assignt assignment(lhs, printf_code);
      assignment.location()=function.location();
      copy(assignment, ASSIGN, dest);
    }
    else
    {
      printf_code.id("code");
      printf_code.type()=typet("code");
      copy(to_code(printf_code), OTHER, dest);
    }
  }
}

/*******************************************************************\

Function: goto_convertt::do_atomic_begin

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_atomic_begin(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(lhs.is_not_nil())
  {
    err_location(lhs);
    throw "atomic_begin does not expect an LHS";
  }

  if(arguments.size() != 0)
  {
    err_location(function);
    throw "atomic_begin takes zero argument";
  }

  goto_programt::targett t=dest.add_instruction(ATOMIC_BEGIN);
  t->location=function.location();
}

/*******************************************************************\

Function: goto_convertt::do_atomic_end

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_atomic_end(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(lhs.is_not_nil())
  {
    err_location(lhs);
    throw "atomic_end does not expect an LHS";
  }

  if(!arguments.empty())
  {
    err_location(function);
    throw "atomic_end takes no arguments";
  }

  goto_programt::targett t=dest.add_instruction(ATOMIC_END);
  t->location=function.location();
}

/*******************************************************************\

Function: goto_convertt::do_malloc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_malloc(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(arguments.size()!=1)
  {
    err_location(function);
    throw "malloc expected to have one argument";
  }

  if(lhs.is_nil())
    return; // does nothing

  locationt location=function.location();

  // get alloc type and size
  typet alloc_type;
  exprt alloc_size;

  get_alloc_type(arguments[0], alloc_type, alloc_size);

  if(alloc_size.is_nil())
    alloc_size=from_integer(1, uint_type());

  if(alloc_type.is_nil())
    alloc_type=char_type();

  if(alloc_size.type()!=uint_type())
  {
    alloc_size.make_typecast(uint_type());
    simplify(alloc_size);
  }

  // produce new object

  exprt new_expr("sideeffect", lhs.type());
  new_expr.statement("malloc");
  new_expr.copy_to_operands(arguments[0]);
  new_expr.cmt_size(alloc_size);
  new_expr.cmt_type(alloc_type);
  new_expr.location()=location;

  goto_programt::targett t_n=dest.add_instruction(ASSIGN);
  t_n->code=code_assignt(lhs, new_expr);
  t_n->location=location;

  exprt lhs_pointer=lhs;
  if(lhs_pointer.type().id()!="pointer")
    lhs_pointer.make_typecast(pointer_typet(empty_typet()));

  // set up some expressions
  exprt valid_expr("valid_object", typet("bool"));
  valid_expr.copy_to_operands(lhs_pointer);
  valid_expr.location()=location;
  exprt neg_valid_expr=gen_not(valid_expr);

  //tse paper
#if TSE_PAPER
  exprt deallocated_expr("deallocated_object", typet("bool"));
  deallocated_expr.copy_to_operands(lhs_pointer);
  deallocated_expr.location()=location;
  exprt neg_deallocated_expr=gen_not(deallocated_expr);
#endif

  exprt pointer_offset_expr("pointer_offset", int_type());
  pointer_offset_expr.location()=location;
  pointer_offset_expr.copy_to_operands(lhs_pointer);

  equality_exprt offset_is_zero_expr(
    pointer_offset_expr, gen_zero(int_type()));

  // first assume that it's available and that it's a dynamic object
  goto_programt::targett t_a=dest.add_instruction(ASSUME);
  t_a->guard=(neg_valid_expr, offset_is_zero_expr);

  // set size
  //nec: ex37.c
  exprt dynamic_size("dynamic_size", int_type()/*uint_type()*/);
  dynamic_size.copy_to_operands(lhs_pointer);
  dynamic_size.location()=location;
  goto_programt::targett t_s_s=dest.add_instruction(ASSIGN);
  t_s_s->code=code_assignt(dynamic_size, alloc_size);
  t_s_s->location=location;

  // now set alloc bit
  goto_programt::targett t_s_a=dest.add_instruction(ASSIGN);
  t_s_a->code=code_assignt(valid_expr, true_exprt());
  t_s_a->location=location;

  //tse paper
#if TSE_PAPER
  //now set deallocated bit
  goto_programt::targett t_d_i=dest.add_instruction(ASSIGN);
  t_d_i->code=code_assignt(deallocated_expr, false_exprt());
  t_d_i->location=location;
#endif

  exprt allocated_object = lhs;
  allocated_object.location() = function.location();

  if (options.get_bool_option("memory-leak-check")
	  && allocated_object.type().id()=="pointer")
    allocated_objects.push(allocated_object);
}

/*******************************************************************\

Function: goto_convertt::do_cpp_new

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_cpp_new(
  exprt &lhs,
  exprt &rhs,
  goto_programt &dest)
{
  if(lhs.is_nil())
  {
    // TODO
    assert(0);
  }

  // grab initializer
  goto_programt tmp_initializer;
  cpp_new_initializer(lhs, rhs, tmp_initializer);

  // produce new object
  goto_programt::targett t_n=dest.add_instruction(ASSIGN);
  t_n->code=code_assignt(lhs, rhs);
  t_n->location=rhs.find_location();

  // set up some expressions
  exprt valid_expr("valid_object", typet("bool"));
  valid_expr.copy_to_operands(lhs);

  // first assume that it's available
  goto_programt::targett t_a=dest.add_instruction(ASSUME);

  t_a->guard=valid_expr;
  t_a->guard.make_not();

  exprt alloc_size;

  if(rhs.statement()=="cpp_new[]")
  {
    alloc_size=static_cast<const exprt &>(rhs.size_irep());
    if(alloc_size.type()!=uint_type())
      alloc_size.make_typecast(uint_type());
  }
  else
    alloc_size=from_integer(1, uint_type());

  // set size
  //nec: ex37.c
  exprt dynamic_size("dynamic_size", int_type()/*uint_type()*/);
  dynamic_size.copy_to_operands(lhs);
  dynamic_size.location()=rhs.find_location();
  goto_programt::targett t_s_s=dest.add_instruction(ASSIGN);
  t_s_s->code=code_assignt(dynamic_size, alloc_size);
  t_s_s->location=rhs.find_location();

  // now set alloc bit
  goto_programt::targett t_s_a=dest.add_instruction(ASSIGN);
  t_s_a->code=code_assignt(valid_expr, true_exprt());
  t_s_a->location=rhs.find_location();

  // run initializer
  dest.destructive_append(tmp_initializer);
}

/*******************************************************************\

Function: goto_convertt::cpp_new_initializer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::cpp_new_initializer(
  const exprt &lhs,
  exprt &rhs,
  goto_programt &dest)
{
  // grab initializer
  exprt initializer;

  if(rhs.initializer().is_nil())
    initializer.make_nil();
  else
  {
    initializer = (exprt&)rhs.initializer();
    rhs.remove("initializer");
  }

  if(initializer.is_not_nil())
  {
    if(rhs.id()=="cpp_new[]")
    {
      // build loop
    }
    else // cpp_new
    {
      exprt deref_new("dereference", rhs.type().subtype());
      deref_new.copy_to_operands(lhs);
      replace_new_object(deref_new, initializer);
      convert(to_code(initializer), dest);
    }
  }
}

/*******************************************************************\

Function: goto_convertt::do_exit

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_exit(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(arguments.size()!=1)
  {
    err_location(function);
    throw "exit expected to have one argument";
  }

  // same as assume(false)

  goto_programt::targett t_a=dest.add_instruction(ASSUME);
  t_a->guard=false_exprt();
  t_a->location=function.location();
}

/*******************************************************************\

Function: goto_convertt::do_abort

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_abort(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(arguments.size()!=0)
  {
    err_location(function);
    throw "abort expected to have no arguments";
  }

  // same as assume(false)

  goto_programt::targett t_a=dest.add_instruction(ASSUME);
  t_a->guard=false_exprt();
  t_a->location=function.location();
}

/*******************************************************************\

Function: goto_convertt::do_array_set

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_array_set(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(arguments.size()!=2)
    throw "array_set expects two arguments";

  const exprt &array_ptr=arguments[0];
  const exprt &value=arguments[1];

  if(array_ptr.id()!="implicit_address_of")
    throw "array_set expects array-pointer as first argument";

  if(!array_ptr.op0().type().is_array())
    throw "array_set expects array as first argument";

  const exprt &array=array_ptr.op0();

  exprt assignment_rhs("array_of", array.type());
  assignment_rhs.copy_to_operands(value);

  codet assignment("assign");

  assignment.reserve_operands(2);
  assignment.copy_to_operands(array);
  assignment.move_to_operands(assignment_rhs);

  convert(assignment, dest);
}

/*******************************************************************\

Function: goto_convertt::do_free

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_free(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(arguments.size()!=1)
  {
    err_location(function);
    throw "free expected to have one argument";
  }

  if(lhs.is_not_nil())
  {
    err_location(function);
    throw "free is expected not to have LHS";
  }

  // preserve the call
  codet free_statement("free");
  free_statement.location()=function.location();
  free_statement.copy_to_operands(arguments[0]);

  goto_programt::targett t_f=dest.add_instruction(OTHER);
  t_f->code=free_statement;
  t_f->location=function.location();

  exprt valid_expr("valid_object", bool_typet());
  valid_expr.copy_to_operands(arguments[0]);

  //tse paper
#if TSE_PAPER
  exprt deallocated_expr("deallocated_object", bool_typet());
  deallocated_expr.copy_to_operands(arguments[0]);
#endif

  // clear alloc bit

  goto_programt::targett t_c=dest.add_instruction(ASSIGN);
  t_c->code=code_assignt(valid_expr, false_exprt());
  t_c->location=function.location();

  //tse paper
#if TSE_PAPER
  //indicate that memory has been deallocated

  goto_programt::targett t_d=dest.add_instruction(ASSIGN);
  t_d->code=code_assignt(deallocated_expr, true_exprt());
  t_d->location=function.location();
#endif
}

/*******************************************************************\

Function: goto_convertt::do_abs

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::do_abs(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(lhs.is_nil()) return;

  if(arguments.size()!=1)
  {
    err_location(function);
    throw "abs expected to have one argument";
  }

  const exprt &arg=arguments.front();

  exprt uminus=exprt("uminus", arg.type());
  uminus.copy_to_operands(arg);

  exprt rhs=exprt("if", arg.type());
  rhs.operands().resize(3);
  rhs.op0()=binary_relation_exprt(arg, ">=", gen_zero(arg.type()));
  rhs.op1()=arg;
  rhs.op2()=uminus;

  code_assignt assignment(lhs, rhs);
  assignment.location()=function.location();
  copy(assignment, ASSIGN, dest);
}

/*******************************************************************\

Function: goto_convertt::do_function_call_symbol

  Inputs:

 Outputs:

 Purpose: add function calls to function queue for later
          processing

\*******************************************************************/

void goto_convertt::do_function_call_symbol(
  const exprt &lhs,
  const exprt &function,
  const exprt::operandst &arguments,
  goto_programt &dest)
{
  if(function.invalid_object())
    return; // ignore

  // lookup symbol
  const irep_idt &identifier=function.identifier();

  const symbolt *symbol;
  if(ns.lookup(identifier, symbol))
  {
    err_location(function);
    throw "error: function `"+id2string(identifier)+"' not found";
  }

  if(!symbol->type.is_code())
  {
    err_location(function);
    throw "error: function `"+id2string(identifier)+"' type mismatch: expected code";
  }

  bool is_assume=identifier==CPROVER_PREFIX "assume";

  bool is_assert=identifier=="c::assert";

  if(is_assume || is_assert)
  {
    if(arguments.size()!=1)
    {
      err_location(function);
      throw "`"+id2string(identifier)+"' expected to have one argument";
    }

    if(options.get_bool_option("no-assertions") && !is_assume)
      return;

    goto_programt::targett t=dest.add_instruction(
      is_assume?ASSUME:ASSERT);
    t->guard=arguments.front();
    t->location=function.location();
    t->location.user_provided(true);

    if(is_assert)
      t->location.property("assertion");

    if(lhs.is_not_nil())
    {
      err_location(function);
      throw id2string(identifier)+" expected not to have LHS";
    }
  }
  else if(identifier==CPROVER_PREFIX "assert")
  {
    if(arguments.size()!=2)
    {
      err_location(function);
      throw "`"+id2string(identifier)+"' expected to have two arguments";
    }

    const std::string &description=
      get_string_constant(arguments[1]);

    if(options.get_bool_option("no-assertions") &&
   	   !(description.find("deadlock detected") != std::string::npos))
      return;

    goto_programt::targett t=dest.add_instruction(ASSERT);
    t->guard=arguments[0];
    t->location=function.location();
    t->location.user_provided(true);
    t->location.property("assertion");
    t->location.comment(description);

    if(lhs.is_not_nil())
    {
      err_location(function);
      throw id2string(identifier)+" expected not to have LHS";
    }
  }
  else if(identifier==CPROVER_PREFIX "printf")
  {
    do_printf(lhs, function, arguments, dest);
  }
  else if(identifier==CPROVER_PREFIX "atomic_begin")
  {
    do_atomic_begin(lhs, function, arguments, dest);
  }
  else if(identifier==CPROVER_PREFIX "atomic_end")
  {
    do_atomic_end(lhs, function, arguments, dest);
  }
  else if(has_prefix(id2string(identifier), "c::nondet_") ||
          has_prefix(id2string(identifier), "cpp::nondet_"))
  {
    // make it a side effect if there is an LHS
    if(lhs.is_nil()) return;

    exprt rhs=side_effect_expr_nondett(lhs.type());
    rhs.location()=function.location();

    code_assignt assignment(lhs, rhs);
    assignment.location()=function.location();
    copy(assignment, ASSIGN, dest);
  }
  else if(has_prefix(id2string(identifier), CPROVER_PREFIX "array_set"))
  {
    do_array_set(lhs, function, arguments, dest);
  }
  else if(identifier=="c::exit")
  {
    do_exit(lhs, function, arguments, dest);
  }
  else if(identifier=="c::abort")
  {
    do_abort(lhs, function, arguments, dest);
  }
  else if(identifier=="c::pthread_create")
  {
    do_pthread_create(lhs, function, arguments, dest);
  }
  else if(identifier=="c::pthread_exit")
  {
    do_pthread_exit(lhs, function, arguments, dest);
  }
  else if(identifier=="c::malloc")
  {
    do_malloc(lhs, function, arguments, dest);
  }
  else if(identifier=="c::free")
  {
    do_free(lhs, function, arguments, dest);
  }
  else if(identifier=="c::printf" ||
          identifier=="c::fprintf" ||
          identifier=="c::sprintf" ||
          identifier=="c::snprintf")
  {
    do_printf(lhs, function, arguments, dest);
  }
  else if(identifier=="c::__assert_rtn" ||
          identifier=="c::__assert_fail")
  {
    // __assert_fail is Linux
    // These take four arguments:
    // "expression", "file.c", line, __func__

    if(arguments.size()!=4)
    {
      err_location(function);
      throw "`"+id2string(identifier)+"' expected to have four arguments";
    }

    std::string description = "assertion ";

    //check whether the assert does not contain a member
    if (arguments[0].id() != "address_of" &&
    	arguments[0].op0().op0().id() != "member") {
    	description = "assertion "+get_string_constant(arguments[0]);
    }

    if(options.get_bool_option("no-assertions"))
      return;

    goto_programt::targett t=dest.add_instruction(ASSERT);
    t->guard=false_exprt();
    t->location=function.location();
    t->location.user_provided(true);
    t->location.property("assertion");
    t->location.comment(description);
    // we ignore any LHS
  }
  else if(identifier=="c::__assert_rtn")
  {
    // __assert_rtn is MACOS

    if(arguments.size()!=4)
    {
      err_location(function);
      throw "`"+id2string(identifier)+"' expected to have four arguments";
    }

    const std::string description=
      "assertion "+get_string_constant(arguments[3]);

    if(options.get_bool_option("no-assertions"))
      return;

    goto_programt::targett t=dest.add_instruction(ASSERT);
    t->guard=false_exprt();
    t->location=function.location();
    t->location.user_provided(true);
    t->location.property("assertion");
    t->location.comment(description);
    // we ignore any LHS
  }
  else if(identifier=="c::_wassert")
  {
    // this is Windows

    if(arguments.size()!=3)
    {
      err_location(function);
      throw "`"+id2string(identifier)+"' expected to have three arguments";
    }

    const std::string description=
      "assertion "+get_string_constant(arguments[0]);

    if(options.get_bool_option("no-assertions"))
      return;

    goto_programt::targett t=dest.add_instruction(ASSERT);
    t->guard=false_exprt();
    t->location=function.location();
    t->location.user_provided(true);
    t->location.property("assertion");
    t->location.comment(description);
    // we ignore any LHS
  }
  else
  {
    do_function_call_symbol(*symbol);

    // insert function call
    code_function_callt function_call;
    function_call.lhs()=lhs;
    function_call.function()=function;
    function_call.arguments()=arguments;
    function_call.location()=function.location();

    copy(function_call, FUNCTION_CALL, dest);
  }
}
