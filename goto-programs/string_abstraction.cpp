/*******************************************************************\

Module: String Abstraction

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <string.h>

#include <std_expr.h>
#include <std_code.h>
#include <expr_util.h>
#include <message_stream.h>
#include <arith_tools.h>
#include <config.h>
#include <pointer_arithmetic.h>
#include <i2string.h>
#include <bitvector.h>

#include <ansi-c/c_types.h>

#include "string_abstraction.h"

/*******************************************************************\

   Class: string_abstractiont

 Purpose:

\*******************************************************************/

class string_abstractiont:public message_streamt
{
public:
  string_abstractiont(
    contextt &_context,
    message_handlert &_message_handler):
    message_streamt(_message_handler),
    context(_context),
    ns(_context)
  {
    struct_typet s;

    s.components().resize(3);

    s.components()[0].set("name", "is_zero");
    s.components()[0].set("pretty_name", "is_zero");
    s.components()[0].type()=build_type(IS_ZERO);

    s.components()[1].set("name", "length");
    s.components()[1].set("pretty_name", "length");
    s.components()[1].type()=build_type(LENGTH);

    s.components()[2].set("name", "size");
    s.components()[2].set("pretty_name", "size");
    s.components()[2].type()=build_type(SIZE);

    string_struct=s;
  }
  
  void operator()(goto_functionst &dest);

  exprt is_zero_string(
    const exprt &object,
    bool write,
    const locationt &location);

  exprt zero_string_length(
    const exprt &object,
    bool write,
    const locationt &location);

  exprt buffer_size(
    const exprt &object,
    const locationt &location);

  static bool has_string_macros(const exprt &expr);

  void replace_string_macros(
    exprt &expr,
    bool lhs,
    const locationt &location);
  
  typet get_string_struct(void) { return string_struct; }

protected:
  contextt &context;
  namespacet ns;

  void move_lhs_arithmetic(exprt &lhs, exprt &rhs);

  bool is_char_type(const typet &type) const
  {
    if(type.id()=="symbol")
      return is_char_type(ns.follow(type));

    if(type.id()!="signedbv" &&
       type.id()!="unsignedbv")
      return false;

    return bv_width(type)==config.ansi_c.char_width;
  }

  void make_type(exprt &dest, const typet &type)
  {
    if(dest.is_not_nil() &&
       ns.follow(dest.type())!=ns.follow(type))
      dest.make_typecast(type);
  }

  void abstract(goto_programt &dest, goto_programt::targett it);
  void abstract_assign(goto_programt &dest, goto_programt::targett it);
  void abstract_pointer_assign(goto_programt &dest, goto_programt::targett it);
  void abstract_char_assign(goto_programt &dest, goto_programt::targett it);
  void abstract_function_call(goto_programt &dest, goto_programt::targett it);

  typedef enum { IS_ZERO, LENGTH, SIZE } whatt;

  exprt build(
    const exprt &pointer,
    whatt what,
    bool write,
    const locationt &location);

  exprt build(const exprt &ptr, bool write);
  exprt build_symbol_ptr(const exprt &object);
  exprt build_symbol_buffer(const exprt &object);
  exprt build_symbol_constant(const irep_idt &str);
  exprt build_unknown(whatt what, bool write);
  exprt build_unknown(bool write);
  static typet build_type(whatt what);

  exprt sub(const exprt &a, const exprt &b)
  {
    if(b.is_nil() || b.is_zero()) return a;
    exprt result("-", a.type());
    result.copy_to_operands(a, b);
    make_type(result.op1(), result.type());
    return result;
  }

  exprt member(const exprt &a, whatt what)
  {
    if(a.is_nil()) return a;
    exprt result("member", build_type(what));
    result.copy_to_operands(a);

    switch(what)
    {
    case IS_ZERO: result.set("component_name", "is_zero"); break;
    case SIZE: result.set("component_name", "size"); break;
    case LENGTH: result.set("component_name", "length"); break;
    default: assert(false);
    }

    return result;
  }

  typet string_struct;
  goto_programt initialization;  

  typedef std::map<irep_idt, irep_idt> localst;
  localst locals;
  
  void abstract(goto_function_templatet<goto_programt> &dest);
};

/*******************************************************************\

Function: string_abstraction

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstraction(
  contextt &context,
  message_handlert &message_handler,
  goto_functionst &dest)
{
  string_abstractiont string_abstraction(context, message_handler);
  string_abstraction(dest);
}

/*******************************************************************\

Function: string_abstractiont::operator()

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstractiont::operator()(goto_functionst &dest)
{
  for(goto_functionst::function_mapt::iterator
      it=dest.function_map.begin();
      it!=dest.function_map.end();
      it++)
    abstract(it->second);

  // do we have a main?
  goto_functionst::function_mapt::iterator
    m_it=dest.function_map.find(dest.main_id());

  if(m_it!=dest.function_map.end())
  {
    goto_programt &main=m_it->second.body;

    // do initialization
    initialization.destructive_append(main);
    main.swap(initialization);
    initialization.clear();
  }
}

/*******************************************************************\

Function: string_abstractiont::abstract

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstractiont::abstract(goto_function_templatet<goto_programt> &dest)
{
  locals.clear();

  code_typet &func_type = static_cast<code_typet &>(dest.type);
  code_typet::argumentst &arg_types=func_type.arguments();
  code_typet::argumentst new_args;

  for(code_typet::argumentst::const_iterator it=arg_types.begin();
      it!=arg_types.end(); it++)
  {
    const code_typet::argumentt &argument=
                static_cast<const code_typet::argumentt &>(*it);
    const typet &type = argument.type();

    new_args.push_back(argument);

    if(type.id()=="pointer" && is_char_type(type.subtype()))
    {
      code_typet::argumentt new_arg;

      new_arg.type() = pointer_typet(string_struct);
      new_arg.set_identifier(it->get_identifier().as_string() + "#str");
      new_arg.set_base_name(it->get_base_name().as_string() + "#str");
      new_args.push_back(new_arg);
    }
  }

  func_type.arguments() = new_args;

  Forall_goto_program_instructions(it, dest.body)
    abstract(dest.body, it);

  // do it again for the locals
  if(!locals.empty())
  {
    Forall_goto_program_instructions(it, dest.body)
    {
      for(localst::const_iterator
          l_it=locals.begin();
          l_it!=locals.end();
          l_it++)
      {
        it->add_local_variable(l_it->second);
      }

      // do initializations of those locals
      if(it->is_other() && it->code.get("statement")=="decl")
      {
        assert(it->code.operands().size()==1);
        if(it->code.op0().id()=="symbol")
        {
          const irep_idt &identifier=
            to_symbol_expr(it->code.op0()).get_identifier();

          localst::const_iterator l_it=locals.find(identifier);
          if(l_it!=locals.end())
          {
            const symbolt &symbol=ns.lookup(l_it->second);

            if(symbol.value.is_not_nil())
            {
              // initialization
              goto_programt tmp;

              goto_programt::targett decl1=tmp.add_instruction();
              decl1->make_other();
              decl1->code=code_declt();
              decl1->code.copy_to_operands(symbol_expr(symbol));
              decl1->location=it->location;
              decl1->local_variables=it->local_variables;

              goto_programt::targett assignment1=tmp.add_instruction(ASSIGN);
              assignment1->code=code_assignt(symbol_expr(symbol), symbol.value);
              assignment1->location=it->location;
              assignment1->local_variables=it->local_variables;

              goto_programt::targett it_next=it;
              it_next++;

              dest.body.destructive_insert(it_next, tmp);
            }
          }
        }
      }
    }
  }

  locals.clear();
}

/*******************************************************************\

Function: string_abstractiont::abstract

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstractiont::abstract(
  goto_programt &dest,
  goto_programt::targett it)
{
  switch(it->type)
  {
  case ASSIGN:
    abstract_assign(dest, it);
    break;
    
  case GOTO:
  case ASSERT:
  case ASSUME:
    if(has_string_macros(it->guard))
      replace_string_macros(it->guard, false, it->location);
    break;
    
  case FUNCTION_CALL:
    abstract_function_call(dest, it);
    break;
    
  case RETURN:
    if(it->code.operands().size()!=0)
      replace_string_macros(it->code.op0(), false, it->location);
    break;
    
  default:;  
  }
}

/*******************************************************************\

Function: string_abstractiont::has_string_macros

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool string_abstractiont::has_string_macros(const exprt &expr)
{
  if(expr.id()=="is_zero_string" ||
     expr.id()=="zero_string_length" ||
     expr.id()=="buffer_size")
    return true;

  forall_operands(it, expr)
    if(has_string_macros(*it))
      return true;

  return false;
}

/*******************************************************************\

Function: string_abstractiont::replace_string_macros

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstractiont::replace_string_macros(
  exprt &expr,
  bool lhs,
  const locationt &location)
{
  if(expr.id()=="is_zero_string")
  {
    assert(expr.operands().size()==1);
    exprt tmp=is_zero_string(expr.op0(), lhs, location);
    expr.swap(tmp);
  }
  else if(expr.id()=="zero_string_length")
  {
    assert(expr.operands().size()==1);
    exprt tmp=zero_string_length(expr.op0(), lhs, location);
    expr.swap(tmp);
  }
  else if(expr.id()=="buffer_size")
  {
    assert(expr.operands().size()==1);
    exprt tmp=buffer_size(expr.op0(), location);
    expr.swap(tmp);
  }
  else
    Forall_operands(it, expr)
      replace_string_macros(*it, lhs, location);
}

/*******************************************************************\

Function: string_abstractiont::build_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

typet string_abstractiont::build_type(whatt what)
{
  typet type;

  switch(what)
  {
  case IS_ZERO: type=bool_typet(); break;
  case LENGTH:  type=uint_type(); break;
  case SIZE:    type=uint_type(); break;
  default: assert(false);
  }

  return type;
}

/*******************************************************************\

Function: string_abstractiont::build_unknown

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt string_abstractiont::build_unknown(whatt what, bool write)
{
  typet type=build_type(what);

  if(write)
    return exprt("NULL-object", type);

  exprt result;

  switch(what)
  {
  case IS_ZERO:
    result=false_exprt();
    break;

  case LENGTH:
  case SIZE:
    result=exprt("sideeffect", type);
    result.set("statement", "nondet");
    break;

  default: assert(false);
  }

  return result;
}

/*******************************************************************\

Function: string_abstractiont::build_unknown

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt string_abstractiont::build_unknown(bool write)
{
  typet type=pointer_typet();
  type.subtype()=string_struct;

  if(write)
    return exprt("NULL-object", type);

  exprt result=exprt("constant", type);
  result.set("value", "NULL");

  return result;
}

/*******************************************************************\

Function: string_abstractiont::build

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt string_abstractiont::build(
  const exprt &pointer,
  whatt what,
  bool write,
  const locationt &location)
{
  // take care of pointer typecasts now
  if(pointer.id()=="typecast")
  {
    // cast from another pointer type?
    assert(pointer.operands().size()==1);
    if(pointer.op0().type().id()!="pointer")
      return build_unknown(what, write);

    // recursive call
    return build(pointer.op0(), what, write, location);
  }

  exprt str_ptr=build(pointer, write);

  exprt deref=dereference_exprt(string_struct);
  deref.op0()=str_ptr;
  deref.location()=location;

  exprt result=member(deref, what);

  if(what==LENGTH || what==SIZE)
  {
    // adjust for offset
    exprt pointer_offset("pointer_offset", uint_type());
    pointer_offset.copy_to_operands(pointer);
    result=sub(result, pointer_offset);
  }

  return result;
}

/*******************************************************************\

Function: string_abstractiont::build_symbol_ptr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt string_abstractiont::build_symbol_ptr(const exprt &object)
{
  std::string suffix="#str";
  const exprt *p=&object;

  while(p->id()=="member")
  {
    suffix="#"+p->get_string("component_name")+suffix;
    assert(p->operands().size()==1);
    p=&(p->op0());
  }

  if(p->id()!="symbol")
    return static_cast<const exprt &>(get_nil_irep());

  const symbol_exprt &expr_symbol=to_symbol_expr(*p);

  const symbolt &symbol=ns.lookup(expr_symbol.get_identifier());
  irep_idt identifier=id2string(symbol.name)+suffix;

  typet type=pointer_typet();
  type.subtype()=string_struct;

  if(context.symbols.find(identifier)==
     context.symbols.end())
  {
    symbolt new_symbol;
    new_symbol.name=identifier;
    new_symbol.mode=symbol.mode;
    new_symbol.type=type;
    new_symbol.is_statevar=true;
    new_symbol.lvalue=true;
    new_symbol.static_lifetime=symbol.static_lifetime;
    new_symbol.pretty_name=id2string(symbol.pretty_name)+suffix;
    new_symbol.module=symbol.module;
    new_symbol.base_name=id2string(symbol.base_name)+suffix;

    context.move(new_symbol);
  }

  const symbolt &str_symbol=ns.lookup(identifier);

  if(!str_symbol.static_lifetime)
    locals[symbol.name]=str_symbol.name;

  return symbol_expr(str_symbol);
}

/*******************************************************************\

Function: string_abstractiont::build

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt string_abstractiont::build(const exprt &pointer, bool write)
{
  // take care of typecasts
  if(pointer.id()=="typecast")
  {
    // cast from another pointer type?
    assert(pointer.operands().size()==1);
    if(pointer.op0().type().id()!="pointer")
      return build_unknown(write);

    // recursive call
    return build(pointer.op0(), write);
  }

  // take care of if
  if(pointer.id()=="if")
  {
    exprt result=if_exprt();

    // recursive call
    result.op0()=pointer.op0();
    result.op1()=build(pointer.op1(), write);
    result.op2()=build(pointer.op2(), write);
    result.type()=result.op1().type();
    return result;
  }

  pointer_arithmetict ptr(pointer);

  if(ptr.pointer.id()=="address_of")
  {
    if(write)
      build_unknown(write);

    assert(ptr.pointer.operands().size()==1);

    if(ptr.pointer.op0().id()=="index")
    {
      assert(ptr.pointer.op0().operands().size()==2);

      const exprt &o=ptr.pointer.op0().op0();

      if(o.id()=="string-constant")
      {
        exprt symbol=build_symbol_constant(o.get("value"));

        if(symbol.is_nil())
          return build_unknown(write);

        exprt address_of("address_of", pointer_typet());
        address_of.type().subtype()=string_struct;
        address_of.copy_to_operands(symbol);

        return address_of;
      }

      exprt symbol=build_symbol_buffer(o);

      if(symbol.is_nil())
        return build_unknown(write);

      exprt address_of("address_of", pointer_typet());
      address_of.type().subtype()=string_struct;
      address_of.copy_to_operands(symbol);

      return address_of;
    }
  }
  else
  {
    exprt result=build_symbol_ptr(ptr.pointer);

    if(result.is_nil())
      return build_unknown(write);

    return result;
  }

  return build_unknown(write);
}

/*******************************************************************\

Function: string_abstractiont::build_symbol_buffer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt string_abstractiont::build_symbol_buffer(const exprt &object)
{
  // first of all, it must be a buffer
  const typet &obj_t=ns.follow(object.type());

  if(obj_t.id()!="array")  
    return static_cast<const exprt &>(get_nil_irep());

  const array_typet &obj_array_type=to_array_type(obj_t);

  // we do buffers, arrays of buffers, and a buffer in a struct

  if(object.id()=="index")
  {
    assert(object.operands().size()==2);

    const typet &t=ns.follow(object.op0().type());

    if(object.op0().id()!="symbol" ||
       t.id()!="array")
      return static_cast<const exprt &>(get_nil_irep());

    const symbol_exprt &expr_symbol=to_symbol_expr(object.op0());

    const symbolt &symbol=ns.lookup(expr_symbol.get_identifier());
    std::string suffix="#str_array";
    irep_idt identifier=id2string(symbol.name)+suffix;

    if(context.symbols.find(identifier)==
       context.symbols.end())
    {
      array_typet new_type;
      new_type.size()=to_array_type(t).size();
      new_type.subtype()=string_struct;

      symbolt new_symbol;
      new_symbol.name=identifier;
      new_symbol.mode=symbol.mode;
      new_symbol.type=new_type;
      new_symbol.is_statevar=true;
      new_symbol.lvalue=true;
      new_symbol.static_lifetime=symbol.static_lifetime;
      new_symbol.pretty_name=id2string(symbol.pretty_name)+suffix;
      new_symbol.module=symbol.module;
      new_symbol.base_name=id2string(symbol.base_name)+suffix;
      new_symbol.value.make_nil();

      {
        exprt struct_expr=exprt("struct", string_struct);
        struct_expr.operands().resize(3);
        struct_expr.op0()=false_exprt();
        struct_expr.op1()=obj_array_type.size();
        make_type(struct_expr.op1(), build_type(SIZE));
        struct_expr.op2()=struct_expr.op1();

        exprt value=exprt("array_of", new_type);
        value.copy_to_operands(struct_expr);
        
        new_symbol.value=value;
      }

      if(symbol.static_lifetime)
      {
        // initialization
        goto_programt::targett assignment1=
          initialization.add_instruction(ASSIGN);
        assignment1->code=code_assignt(symbol_expr(new_symbol), new_symbol.value);
      }

      context.move(new_symbol);
    }

    const symbolt &str_array_symbol=ns.lookup(identifier);

    if(!str_array_symbol.static_lifetime)
      locals[symbol.name]=str_array_symbol.name;

    index_exprt result;
    result.array()=symbol_expr(str_array_symbol);
    result.index()=object.op1();
    result.type()=string_struct;

    return result;
  }

  // possibly walk over some members

  std::string suffix="#str";
  const exprt *p=&object;

  while(p->id()=="member")
  {
    suffix="#"+p->get_string("component_name")+suffix;
    assert(p->operands().size()==1);
    p=&(p->op0());
  }

  if(p->id()!="symbol")
    return static_cast<const exprt &>(get_nil_irep());

  const symbol_exprt &expr_symbol=to_symbol_expr(*p);

  const symbolt &symbol=ns.lookup(expr_symbol.get_identifier());
  irep_idt identifier=id2string(symbol.name)+suffix;

  if(context.symbols.find(identifier)==
     context.symbols.end())
  {
    symbolt new_symbol;
    new_symbol.name=identifier;
    new_symbol.mode=symbol.mode;
    new_symbol.type=string_struct;
    new_symbol.is_statevar=true;
    new_symbol.lvalue=true;
    new_symbol.static_lifetime=symbol.static_lifetime;
    new_symbol.pretty_name=id2string(symbol.pretty_name)+suffix;
    new_symbol.module=symbol.module;
    new_symbol.base_name=id2string(symbol.base_name)+suffix;

    {    
      exprt value=exprt("struct", string_struct);
      value.operands().resize(3);
      value.op0()=false_exprt();
      value.op1()=obj_array_type.size();
      make_type(value.op1(), build_type(SIZE));
      value.op2()=value.op1();
      
      new_symbol.value=value;
    }

    if(symbol.static_lifetime)
    {
      // initialization
      goto_programt::targett assignment1=initialization.add_instruction(ASSIGN);
      assignment1->code=code_assignt(symbol_expr(new_symbol), new_symbol.value);
    }

    context.move(new_symbol);
  }

  const symbolt &str_symbol=ns.lookup(identifier);

  if(!str_symbol.static_lifetime)
    locals[symbol.name]=str_symbol.name;

  return symbol_expr(str_symbol);
}

/*******************************************************************\

Function: string_abstractiont::build_symbol_constant

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt string_abstractiont::build_symbol_constant(const irep_idt &str)
{
  unsigned l=strlen(str.c_str());
  irep_idt base="string_constant_str_"+i2string(l);
  irep_idt identifier="string_abstraction::"+id2string(base);

  if(context.symbols.find(identifier)==
     context.symbols.end())
  {
    symbolt new_symbol;
    new_symbol.name=identifier;
    new_symbol.mode="C";
    new_symbol.type=string_struct;
    new_symbol.is_statevar=true;
    new_symbol.lvalue=true;
    new_symbol.static_lifetime=true;
    new_symbol.pretty_name=base;
    new_symbol.base_name=base;

    {
      exprt value=exprt("struct", string_struct);
      value.operands().resize(3);

      value.op0()=true_exprt();
      value.op1()=from_integer(l, build_type(LENGTH));
      value.op2()=from_integer(l+1, build_type(SIZE));

      // initialization
      goto_programt::targett assignment1=initialization.add_instruction(ASSIGN);
      assignment1->code=code_assignt(symbol_expr(new_symbol), value);
    }

    context.move(new_symbol);
  }

  symbol_exprt symbol_expr;
  symbol_expr.type()=string_struct;
  symbol_expr.set_identifier(identifier);

  return symbol_expr;
}

/*******************************************************************\

Function: string_abstractiont::is_zero_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt string_abstractiont::is_zero_string(
  const exprt &object,
  bool write,
  const locationt &location)
{
  return build(object, IS_ZERO, write, location);
}

/*******************************************************************\

Function: string_abstractiont::zero_string_length

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt string_abstractiont::zero_string_length(
  const exprt &object,
  bool write,
  const locationt &location)
{
  return build(object, LENGTH, write, location);
}

/*******************************************************************\

Function: string_abstractiont::buffer_size

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt string_abstractiont::buffer_size(
  const exprt &object,
  const locationt &location)
{
  return build(object, SIZE, false, location);
}

/*******************************************************************\

Function: string_abstractiont::move_lhs_arithmetic

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstractiont::move_lhs_arithmetic(exprt &lhs, exprt &rhs)
{
  if(lhs.id()=="-")
  {
    // move op1 to rhs
    exprt rest=lhs.op0();
    exprt sum=exprt("+", lhs.type());
    sum.copy_to_operands(rhs, lhs.op1());
    // overwrite
    rhs=sum;
    lhs=rest;
  }
}

/*******************************************************************\

Function: string_abstractiont::abstract_assign

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstractiont::abstract_assign(
  goto_programt &dest,
  goto_programt::targett target)
{
  code_assignt &assign=to_code_assign(target->code);

  exprt &lhs=assign.lhs();
  exprt &rhs=assign.rhs();

  if(has_string_macros(lhs))
  {
    replace_string_macros(lhs, true, target->location);
    move_lhs_arithmetic(lhs, rhs);
  }

  if(has_string_macros(rhs))
    replace_string_macros(rhs, false, target->location);

  if(lhs.type().id()=="pointer")
    abstract_pointer_assign(dest, target);
  else if(is_char_type(lhs.type()))
    abstract_char_assign(dest, target);
}

/*******************************************************************\

Function: string_abstractiont::abstract_pointer_assign

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstractiont::abstract_pointer_assign(
  goto_programt &dest,
  goto_programt::targett target)
{
  code_assignt &assign=to_code_assign(target->code);

  exprt &lhs=assign.lhs();
  exprt rhs=assign.rhs();
  exprt *rhsp=&(assign.rhs());

  while(rhsp->id()=="typecast")
    rhsp=&(rhsp->op0());
  
  // we only care about char pointers for now
  if(!is_char_type(rhsp->type().subtype()))
    return;

  // assign length and is_zero as well

  goto_programt tmp;

  goto_programt::targett assignment=tmp.add_instruction(ASSIGN);
  assignment->code=code_assignt(build(lhs, true), build(rhs, false));
  assignment->location=target->location;
  assignment->local_variables=target->local_variables;

  target++;
  dest.destructive_insert(target, tmp);
}

/*******************************************************************\

Function: string_abstractiont::abstract_char_assign

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstractiont::abstract_char_assign(
  goto_programt &dest,
  goto_programt::targett target)
{
  code_assignt &assign=to_code_assign(target->code);

  exprt &lhs=assign.lhs();
  exprt &rhs=assign.rhs();

  // we only care if the constant zero is assigned
  if(!rhs.is_zero())
    return;

  if(lhs.id()=="index")
  {
    assert(lhs.operands().size()==2);

    goto_programt tmp;

    const exprt symbol_buffer=build_symbol_buffer(lhs.op0());

    const exprt i1=member(symbol_buffer, IS_ZERO);
    if(i1.is_not_nil())
    {
      goto_programt::targett assignment1=tmp.add_instruction(ASSIGN);
      assignment1->code=code_assignt(i1, true_exprt());
      assignment1->code.location()=target->location;
      assignment1->location=target->location;
      assignment1->local_variables=target->local_variables;
    }

    const exprt i2=member(symbol_buffer, LENGTH);
    if(i2.is_not_nil())
    {
      exprt new_length=lhs.op1();
      make_type(new_length, i2.type());

      if_exprt min_expr;
      min_expr.cond()=binary_relation_exprt(new_length, "<", i2);
      min_expr.true_case()=new_length;
      min_expr.false_case()=i2;
      min_expr.type()=i2.type();

      goto_programt::targett assignment2=tmp.add_instruction(ASSIGN);
      assignment2->code=code_assignt(i2, min_expr);
      assignment2->code.location()=target->location;
      assignment2->location=target->location;
      assignment2->local_variables=target->local_variables;

      move_lhs_arithmetic(
       assignment2->code.op0(),
       assignment2->code.op1());
    }

    target++;
    dest.destructive_insert(target, tmp);
  }
  else if(lhs.id()=="dereference")
  {
    assert(lhs.operands().size()==1);

  }
}

/*******************************************************************\

Function: string_abstractiont::abstract_function_call

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstractiont::abstract_function_call(
  goto_programt &dest,
  goto_programt::targett target)
{
  const code_function_callt &call=to_code_function_call(target->code);
  const code_function_callt::argumentst &arguments=call.arguments();
  
  symbolst::const_iterator f_it = 
    context.symbols.find(call.function().get("identifier"));
  if(f_it==context.symbols.end())
    throw "invalid function call";
  
  const code_typet &fnc_type = 
    static_cast<const code_typet &>(f_it->second.type);
  const code_typet::argumentst &argument_types=fnc_type.arguments();
  
  exprt::operandst::const_iterator it1=arguments.begin();

  for(code_typet::argumentst::const_iterator it2=argument_types.begin();
      it2!=argument_types.end();
      it2++)
  {
    if(it1==arguments.end())
    {
      err_location(target->location);
      throw "function call: not enough arguments";
    }
    
    const exprt &argument=static_cast<const exprt &>(*it2);
    
    const exprt actual(*it1);
    
    const exprt *tcfree = &actual;
    while(tcfree->id()=="typecast")
      tcfree=&tcfree->op0();
    
    if(tcfree->type().id()=="pointer" &&
       is_char_type(tcfree->type().subtype()))
    {
      const irep_idt &identifier=argument.get("#identifier");
      
      if(identifier=="") continue; // just forget it
      
      goto_programt tmp;
      
      goto_programt::targett assignment=tmp.add_instruction(ASSIGN);
      assignment->code=code_assignt(
                         build(symbol_exprt(identifier, argument.type()), true), 
                         build(actual, false));
      assignment->location=target->location;
      assignment->local_variables=target->local_variables;
      assignment->function=call.function().get("identifier");

      // inserts _before_ the call
      dest.destructive_insert(target, tmp);
    }
    
    it1++;
  }
}
