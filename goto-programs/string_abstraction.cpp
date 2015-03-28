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
    s.tag("#!strabs_tag");

    s.components()[0].name("is_zero");
    s.components()[0].pretty_name("is_zero");
    s.components()[0].type()=build_type(IS_ZERO);

    s.components()[1].name("length");
    s.components()[1].pretty_name("length");
    s.components()[1].type()=build_type(LENGTH);

    s.components()[2].name("size");
    s.components()[2].pretty_name("size");
    s.components()[2].type()=build_type(SIZE);

    migrate_type(s, string_struct);
  }
  
  void operator()(goto_functionst &dest);

  expr2tc is_zero_string(
    const expr2tc &object,
    bool write,
    const locationt &location);

  expr2tc zero_string_length(
    const expr2tc &object,
    bool write,
    const locationt &location);

  expr2tc buffer_size(
    const expr2tc &object,
    const locationt &location);

  static bool has_string_macros(const expr2tc &expr);

  void replace_string_macros(
    expr2tc &expr,
    bool lhs,
    const locationt &location);
  
  type2tc get_string_struct(void) { return string_struct; }

protected:
  contextt &context;
  namespacet ns;

  void move_lhs_arithmetic(expr2tc &lhs, expr2tc &rhs);

  bool is_char_type(const type2tc &type) const
  {
    if (!is_bv_type(type))
      return false;

    return type->get_width()==config.ansi_c.char_width;
  }

  bool is_char_type(const expr2tc &e) const
  {
    return is_char_type(e->type);
  }

  void make_type(expr2tc &dest, const type2tc &type)
  {
    if (!is_nil_expr(dest) && dest->type != type)
      dest = typecast2tc(type, dest);
  }

  void abstract(irep_idt name, goto_programt &dest, goto_programt::targett it);
  void abstract_assign(goto_programt &dest, goto_programt::targett it);
  void abstract_pointer_assign(goto_programt &dest, goto_programt::targett it);
  void abstract_char_assign(goto_programt &dest, goto_programt::targett it);
  void abstract_function_call(goto_programt &dest, goto_programt::targett it);
  void abstract_return(irep_idt name, goto_programt &dest,
                       goto_programt::targett it);

  typedef enum { IS_ZERO, LENGTH, SIZE } whatt;

  expr2tc build(
    const expr2tc &pointer,
    whatt what,
    bool write,
    const locationt &location);

  expr2tc build(const expr2tc &ptr, bool write);
  expr2tc build_symbol_ptr(const expr2tc &object);
  expr2tc build_symbol_buffer(const expr2tc &object);
  expr2tc build_symbol_constant(const irep_idt &str);
  expr2tc build_unknown(whatt what, bool write);
  expr2tc build_unknown(bool write);
  static typet build_type(whatt what);

  expr2tc sub(const expr2tc &a, const expr2tc &b)
  {
    if (is_nil_expr(b) ||
        (is_constant_bool2t(b) && to_constant_bool2t(b).constant_value))
      return a;

    expr2tc b2 = b;
    make_type(b2, a->type);
    sub2tc res(a->type, a, b2);
    return res;
  }

  expr2tc member(const expr2tc &a, whatt what)
  {
    if (is_nil_expr(a))
      return a;

    irep_idt name;

    switch(what) {
    case IS_ZERO: name = "is_zero"; break;
    case SIZE: name = "size"; break;
    case LENGTH: name = "length"; break;
    default: assert(false);
    }

    type2tc type;
    typet tmp_type = build_type(what);
    migrate_type(tmp_type, type);
    member2tc result(type, a, name);

    return result;
  }

  type2tc string_struct;
  goto_programt initialization;  

  typedef std::map<irep_idt, irep_idt> localst;
  localst locals;

  // Counter numbering the returns in a function. Required for distinguishing
  // labels we may add when altering control flow around returns. Specifically,
  // when assigning string struct pointer of a returned string pointer back to
  // the calling function.
  unsigned int func_return_num;
  
  void abstract(irep_idt name, goto_functiont &dest);
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
    abstract(it->first, it->second);

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

void string_abstractiont::abstract(irep_idt name, goto_functiont &dest)
{
  locals.clear();
  func_return_num = 0;

  type2tc func_type_c;
  migrate_type(dest.type, func_type_c);
  code_type2t &func_type = to_code_type(func_type_c);
  std::vector<type2tc> &arg_types = func_type.arguments;
  std::vector<type2tc> new_args;
  std::vector<irep_idt> new_arg_names;

  unsigned int name_idx = 0;
  for (std::vector<type2tc>::const_iterator it = arg_types.begin();
      it != arg_types.end(); it++, name_idx++)
  {
    const type2tc &type = *it;

    new_args.push_back(type);
    new_arg_names.push_back(func_type.argument_names[name_idx]);

    if (is_pointer_type(type) && is_char_type(to_pointer_type(type).subtype))
    {
      new_args.push_back(type2tc(new pointer_type2t(string_struct)));
      new_arg_names.push_back(func_type.argument_names[name_idx].as_string() +
                              "\\str");

      // We also need to put this new argument into the symbol table.

      // Fabricate a base name.
      size_t endpos = new_arg_names.back().as_string().rfind("::");
      std::string basename = new_arg_names.back().as_string().substr(endpos+2);
      symbolt new_sym;
      new_sym.type = migrate_type_back(new_args.back());
      new_sym.value = exprt();
      new_sym.location = locationt();
      new_sym.location.set_file("<added_by_string_abstraction>");
      new_sym.name = new_arg_names.back();
      new_sym.base_name = basename;
      context.add(new_sym);
    }
  }

  // Additionally, if the return type is a char *, then the func needs to be
  // able to provide related information about the returned string. To implement
  // this, another pointer to a string struct is tacked onto the end of the
  // function arguments.
  const type2tc ret_type = func_type.ret_type;
  if (is_pointer_type(ret_type) &&
      is_char_type(to_pointer_type(ret_type).subtype)) {
    code_typet::argumentt new_arg;

    type2tc fintype = type2tc(new pointer_type2t(type2tc(new pointer_type2t(string_struct))));
    new_args.push_back(fintype);
    new_arg_names.push_back(name.as_string() + "::__strabs::returned_str\\str");

    symbolt new_sym;
    new_sym.type = migrate_type_back(fintype);
    new_sym.value = exprt();
    new_sym.location = locationt();
    new_sym.location.set_file("<added_by_string_abstraction>");
    new_sym.name = new_arg_names.back();
    size_t endpos = new_sym.name.as_string().rfind("::");
    std::string basename = new_sym.name.as_string().substr(endpos+2);
    new_sym.base_name = basename;
    context.add(new_sym);

    new_sym.name = name.as_string() + "::__strabs::returned_str";
    new_sym.base_name = "returned_str";
    new_sym.type = pointer_typet(signedbv_typet(8));
    context.add(new_sym);

    locals[new_sym.name] = new_arg_names.back();
  }

  func_type.arguments = new_args;
  func_type.argument_names = new_arg_names;

  // Additionally, update the type of our symbol
  symbolst::iterator it = context.symbols.find(name);
  assert(it != context.symbols.end());
  it->second.type = migrate_type_back(func_type_c);

  // Also need to update the type in the goto_function struct, not just the
  // symbol table.
  dest.type = to_code_type(it->second.type);

  Forall_goto_program_instructions(it, dest.body)
    abstract(name, dest.body, it);

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
      if (it->is_other() && is_code_decl2t(it->code))
      {
        const code_decl2t &decl = to_code_decl2t(it->code);

        const irep_idt &identifier = decl.value;

        localst::const_iterator l_it=locals.find(identifier);
        if(l_it!=locals.end())
        {
          const symbolt &symbol=ns.lookup(l_it->second);

          if (symbol.value.is_not_nil())
          {
            // initialization
            goto_programt tmp;

            goto_programt::targett decl1=tmp.add_instruction();
            decl1->make_other();
            type2tc new_type;
            migrate_type(symbol.type, new_type);
            decl1->code = code_decl2tc(new_type, symbol.name);
            decl1->location=it->location;
            decl1->local_variables=it->local_variables;

            goto_programt::targett assignment1=tmp.add_instruction(ASSIGN);
            exprt sym = symbol_expr(symbol);
            expr2tc new_sym;
            migrate_expr(sym, new_sym);
            expr2tc val;
            migrate_expr(symbol.value, val);
            assignment1->code = code_assign2tc(new_sym, val);
            assignment1->location=it->location;
            assignment1->local_variables=it->local_variables;

            goto_programt::targett it_next=it;
            it_next++;

            dest.body.destructive_insert(it_next, tmp);
          } else if (symbol.type.id() == "pointer" &&
                     symbol.type.subtype() == migrate_type_back(string_struct)){
            goto_programt tmp;

            type2tc sym_type;
            migrate_type(symbol.type, sym_type);
            expr2tc null = symbol2tc(sym_type, "NULL");

            goto_programt::targett decl1=tmp.add_instruction();
            decl1->make_other();
            type2tc new_type;
            migrate_type(symbol.type, new_type);
            decl1->code = code_decl2tc(new_type, symbol.name);
            decl1->location=it->location;
            decl1->local_variables=it->local_variables;

            goto_programt::targett assignment1=tmp.add_instruction(ASSIGN);
            exprt sym = symbol_expr(symbol);
            expr2tc new_sym;
            migrate_expr(sym, new_sym);
            assignment1->code = code_assign2tc(new_sym, null);
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

  locals.clear();
}

/*******************************************************************\

Function: string_abstractiont::abstract

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstractiont::abstract(
  irep_idt name,
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
    if (has_string_macros(it->guard))
      replace_string_macros(it->guard, false, it->location);
    break;
    
  case FUNCTION_CALL:
    abstract_function_call(dest, it);
    break;
    
  case RETURN:
    abstract_return(name, dest, it);
    break;
    
  default:;  
  }
}

void string_abstractiont::abstract_return(irep_idt name, goto_programt &dest,
                                          goto_programt::targett it)
{
  irep_idt label;

  code_return2t &ret = to_code_return2t(it->code);

  if (is_nil_expr(ret.operand))
    return; // We're not interested at all.

  replace_string_macros(ret.operand, false, it->location);

  expr2tc ret_val = ret.operand;
  while (is_typecast2t(ret_val))
    ret_val = to_typecast2t(ret_val).from;

  if (!is_pointer_type(ret_val) ||
      !is_char_type(to_pointer_type(ret_val->type).subtype))
    return;

  // We have a return type that needs to also write to the callers string
  // struct. Insert some assignments that perform this.

  // However, don't assign to NULL - which will have been passed in if the
  // caller discards the return value
  std::stringstream ss;
  ss << "strabs_ret_str_" << func_return_num++;
  label = irep_idt(ss.str());
  it->labels.push_back(label);

  goto_programt tmp;
  goto_programt::targett branch, assignment;

  type2tc rtype = type2tc(new pointer_type2t(string_struct));
  type2tc rtype2 = type2tc(new pointer_type2t(rtype));
  exprt old_ret_sym =
    symbol_exprt(name.as_string() + "::__strabs::returned_str\\str",
                 migrate_type_back(rtype2));
  expr2tc ret_sym;
  migrate_expr(old_ret_sym, ret_sym);

  // For the purposes of comparing the pointer against NULL, we need to typecast
  // it: other goto convert functions rewrite the returned_str\\str pointer to
  // be a particular pointer (value set foo). Upon which it becomes another type
  typecast2tc cast(rtype, ret_sym);
  symbol2tc null(rtype2, "NULL");
  equality2tc guard(cast, null);

  branch = tmp.add_instruction(GOTO);
  branch->make_goto();
  branch->guard = guard;
  branch->targets.push_back(it);
  branch->location = it->location;
  branch->local_variables = it->local_variables;
  dest.destructive_insert(it, tmp);

  type2tc deref_type = type2tc(new pointer_type2t(string_struct));
  dereference2tc lhs(deref_type, ret_sym);
  expr2tc rhs = build(ret_val, false);
  assignment = tmp.add_instruction(ASSIGN);
  assignment->code = code_assign2tc(lhs, rhs);
  assignment->location = it->location;
  assignment->local_variables = it->local_variables;
  assignment->guard = true_expr;
  dest.destructive_insert(it, tmp);

  return;
}

/*******************************************************************\

Function: string_abstractiont::has_string_macros

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool string_abstractiont::has_string_macros(const expr2tc &expr)
{
  if (is_zero_string2t(expr) || is_zero_length_string2t(expr) ||
      is_buffer_size2t(expr))
    return true;

  forall_operands2(it, idx, expr)
    if (!is_nil_expr(*it) && has_string_macros(*it))
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
  expr2tc &expr,
  bool lhs,
  const locationt &location)
{
  if (is_zero_string2t(expr))
  {
    const zero_string2t &ref = to_zero_string2t(expr);
    expr2tc tmp = is_zero_string(ref.string, lhs, location);
    expr = tmp;
  }
  else if (is_zero_length_string2t(expr))
  {
    const zero_length_string2t &ref = to_zero_length_string2t(expr);
    expr2tc tmp = zero_string_length(ref.string, lhs, location);
    expr = tmp;
  }
  else if (is_buffer_size2t(expr))
  {
    const buffer_size2t &ref = to_buffer_size2t(expr);
    expr2tc tmp = buffer_size(ref.value, location);
    expr = tmp;
  }
  else
  {
    Forall_operands2(it, idx, expr)
      if (!is_nil_expr(*it))
        replace_string_macros(*it, lhs, location);
  }
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

expr2tc string_abstractiont::build_unknown(whatt what, bool write)
{
  typet type = build_type(what);
  type2tc tmp_type;
  migrate_type(type, tmp_type);

  if (write)
    return null_object2tc(tmp_type);

  expr2tc result;

  switch(what)
  {
  case IS_ZERO:
    result = false_expr;
    break;

  case LENGTH:
  case SIZE:
    {
    std::vector<expr2tc> args;
    result = sideeffect2tc(tmp_type, expr2tc(), expr2tc(), args, type2tc(),
                           sideeffect2t::nondet);
    break;
    }

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

expr2tc string_abstractiont::build_unknown(bool write)
{
  type2tc type = type2tc(new pointer_type2t(string_struct));

  if (write)
    return null_object2tc(type);

  symbol2tc result(type, "NULL");
  return result;
}

/*******************************************************************\

Function: string_abstractiont::build

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

expr2tc string_abstractiont::build(
  const expr2tc &pointer,
  whatt what,
  bool write,
  const locationt &location)
{
  // take care of pointer typecasts now
  if (is_typecast2t(pointer))
  {
    // cast from another pointer type?
    if (!is_pointer_type(to_typecast2t(pointer).from))
      return build_unknown(what, write);

    // recursive call
    return build(to_typecast2t(pointer).from, what, write, location);
  }

  expr2tc str_ptr = build(pointer, write);

  dereference2tc deref(string_struct, str_ptr);
  expr2tc result = member(deref, what);

  if (what==LENGTH || what==SIZE)
  {
    // adjust for offset
    pointer_offset2tc ptr_offs(pointer_type2(), pointer);
    result = sub(result, ptr_offs);
  }

  return result;
}

/*******************************************************************\

Function: string_abstractiont::build_symbol_ptr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

expr2tc string_abstractiont::build_symbol_ptr(const expr2tc &object)
{
  std::string suffix="\\str";
  expr2tc obj = object;

  while (is_member2t(obj))
  {
    suffix="\\" + to_member2t(obj).member.as_string() + suffix;
    obj = to_member2t(obj).source_value;
  }

  if (!is_symbol2t(obj) || to_symbol2t(obj).get_symbol_name() == "NULL")
    return expr2tc();

  const symbol2t &expr_symbol = to_symbol2t(obj);

  const symbolt &symbol = ns.lookup(expr_symbol.get_symbol_name());
  irep_idt identifier = symbol.name.as_string() + suffix;

  type2tc type = type2tc(new pointer_type2t(string_struct));

  if(context.symbols.find(identifier) == context.symbols.end())
  {
    symbolt new_symbol;
    new_symbol.name=identifier;
    new_symbol.mode=symbol.mode;
    new_symbol.type = migrate_type_back(type);
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

  exprt sym_exp = symbol_expr(str_symbol);
  expr2tc tmp;
  migrate_expr(sym_exp, tmp);
  return tmp;
}

/*******************************************************************\

Function: string_abstractiont::build

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

expr2tc string_abstractiont::build(const expr2tc &pointer, bool write)
{
  // take care of typecasts
  if (is_typecast2t(pointer))
  {
    // cast from another pointer type?
    if (!is_pointer_type(to_typecast2t(pointer).from))
      return build_unknown(write);

    // recursive call
    return build(to_typecast2t(pointer).from, write);
  }

  // take care of if
  if (is_if2t(pointer))
  {
    const if2t &ifval = to_if2t(pointer);
    expr2tc true_exp = build(ifval.true_value, write);
    expr2tc false_exp = build(ifval.false_value, write);

    // recursive call
    if2tc result(true_exp->type, ifval.cond, true_exp, false_exp);
    return result;
  }

  // migration erk.
  exprt old_pointer = migrate_expr_back(pointer);
  pointer_arithmetict ptr(old_pointer);

  if(ptr.pointer.is_address_of())
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
        exprt symbol = migrate_expr_back(build_symbol_constant(o.value()));

        if(symbol.is_nil())
          return build_unknown(write);

        exprt address_of("address_of", pointer_typet());
        address_of.type().subtype()=migrate_type_back(string_struct);
        address_of.copy_to_operands(symbol);

        expr2tc new_addr_of;
        migrate_expr(address_of, new_addr_of);
        return new_addr_of;
      }

      expr2tc obj;
      migrate_expr(o, obj);
      exprt symbol=migrate_expr_back(build_symbol_buffer(obj));

      if(symbol.is_nil())
        return build_unknown(write);

      exprt address_of("address_of", pointer_typet());
      address_of.type().subtype()=migrate_type_back(string_struct);
      address_of.copy_to_operands(symbol);

      expr2tc new_addr_of;
      migrate_expr(address_of, new_addr_of);
      return new_addr_of;
    }
  }
  else
  {
    expr2tc tmp_ptr;
    migrate_expr(ptr.pointer, tmp_ptr);
    exprt result=migrate_expr_back(build_symbol_ptr(tmp_ptr));

    if(result.is_nil())
      return build_unknown(write);

    expr2tc new_res;
    migrate_expr(result, new_res);
    return new_res;
  }

  return build_unknown(write);
}

/*******************************************************************\

Function: string_abstractiont::build_symbol_buffer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

expr2tc string_abstractiont::build_symbol_buffer(const expr2tc &object)
{
  // first of all, it must be a buffer
  const type2tc &obj_t = object->type;

  if(!is_array_type(obj_t))
    return expr2tc();

  const array_type2t &obj_array_type = to_array_type(obj_t);

  // we do buffers, arrays of buffers, and a buffer in a struct

  if (is_index2t(object))
  {
    const index2t &idx = to_index2t(object);
    const type2tc &t = idx.source_value->type;

    if (!is_symbol2t(idx.source_value) || !is_array_type(t))
      return expr2tc();

    const symbol2t &expr_symbol = to_symbol2t(idx.source_value);
    const array_type2t &arr_type = to_array_type(expr_symbol.type);

    const symbolt &symbol = ns.lookup(expr_symbol.get_symbol_name());
    std::string suffix="\\str_array";
    irep_idt identifier=id2string(symbol.name)+suffix;

    if(context.symbols.find(identifier)==
       context.symbols.end())
    {
      type2tc new_type = type2tc(new array_type2t(string_struct,
                                                  arr_type.array_size,
                                                  arr_type.size_is_infinite));

      symbolt new_symbol;
      new_symbol.name=identifier;
      new_symbol.mode=symbol.mode;
      new_symbol.type = migrate_type_back(new_type);
      new_symbol.is_statevar=true;
      new_symbol.lvalue=true;
      new_symbol.static_lifetime=symbol.static_lifetime;
      new_symbol.pretty_name=id2string(symbol.pretty_name)+suffix;
      new_symbol.module=symbol.module;
      new_symbol.base_name=id2string(symbol.base_name)+suffix;
      new_symbol.value.make_nil();

      {
        std::vector<expr2tc> operands;
        operands.push_back(false_expr);
        operands.push_back(obj_array_type.array_size);
        typet blah = build_type(SIZE);
        type2tc an_op_type;
        migrate_type(blah, an_op_type);
        make_type(operands.back(), an_op_type);
        operands.push_back(operands.back());
        constant_struct2tc struct_expr(string_struct, operands);

        constant_array_of2tc value(new_type, struct_expr);
        
        new_symbol.value = migrate_expr_back(value);
      }

      if(symbol.static_lifetime)
      {
        // initialization
        goto_programt::targett assignment1=
          initialization.add_instruction(ASSIGN);
        exprt sym_exp = symbol_expr(new_symbol);
        expr2tc sym, val;
        migrate_expr(sym_exp, sym);
        migrate_expr(new_symbol.value, val);
        assignment1->code = code_assign2tc(sym, val);
      }

      context.move(new_symbol);
    }

    const symbolt &str_array_symbol=ns.lookup(identifier);

    if(!str_array_symbol.static_lifetime)
      locals[symbol.name]=str_array_symbol.name;

    exprt sym = symbol_expr(str_array_symbol);
    expr2tc sym_exp;
    migrate_expr(sym, sym_exp);
    return index2tc(string_struct, sym_exp, idx.index);
  }

  // possibly walk over some members

  std::string suffix="\\str";
  expr2tc p = object;

  while (is_member2t(p))
  {
    suffix="\\" + to_member2t(p).member.as_string() + suffix;
    p = to_member2t(p).source_value;
  }

  if (!is_symbol2t(p))
    return expr2tc();

  const symbol2t thesym = to_symbol2t(p);

  const symbolt &symbol = ns.lookup(thesym.get_symbol_name());
  irep_idt identifier=id2string(symbol.name) + suffix;

  if(context.symbols.find(identifier)==
     context.symbols.end())
  {
    symbolt new_symbol;
    new_symbol.name=identifier;
    new_symbol.mode=symbol.mode;
    new_symbol.type = migrate_type_back(string_struct);
    new_symbol.is_statevar=true;
    new_symbol.lvalue=true;
    new_symbol.static_lifetime=symbol.static_lifetime;
    new_symbol.pretty_name=id2string(symbol.pretty_name)+suffix;
    new_symbol.module=symbol.module;
    new_symbol.base_name=id2string(symbol.base_name)+suffix;

    {    
      std::vector<expr2tc> operands;
      operands.push_back(false_expr);
      operands.push_back(obj_array_type.array_size);
      typet tmptype = build_type(SIZE);
      type2tc eventmpertype;
      migrate_type(tmptype, eventmpertype);
      make_type(operands.back(), eventmpertype);
      operands.push_back(operands.back());
      constant_struct2tc value(string_struct, operands);
      
      new_symbol.value = migrate_expr_back(value);
    }

    exprt new_sym = symbol_expr(new_symbol);
    exprt new_sym_val = new_symbol.value;
    context.move(new_symbol);

    if(symbol.static_lifetime)
    {
      // initialization
      goto_programt::targett assignment1=initialization.add_instruction(ASSIGN);
      expr2tc new_sym2, new_sym_value;
      migrate_expr(new_sym, new_sym2);
      migrate_expr(new_sym_val, new_sym_value);
      assignment1->code = code_assign2tc(new_sym2, new_sym_value);
    }
  }

  const symbolt &str_symbol=ns.lookup(identifier);

  if(!str_symbol.static_lifetime)
    locals[symbol.name]=str_symbol.name;

  exprt symsymbol = symbol_expr(str_symbol);
  expr2tc newsymsymbol;
  migrate_expr(symsymbol, newsymsymbol);
  return newsymsymbol;
}

/*******************************************************************\

Function: string_abstractiont::build_symbol_constant

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

expr2tc string_abstractiont::build_symbol_constant(const irep_idt &str)
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
    new_symbol.type = migrate_type_back(string_struct);
    new_symbol.is_statevar=true;
    new_symbol.lvalue=true;
    new_symbol.static_lifetime=true;
    new_symbol.pretty_name=base;
    size_t endpos = base.as_string().rfind("::");
    std::string basename = base.as_string().substr(endpos+2);
    new_symbol.base_name=base;

    exprt new_sym = symbol_expr(new_symbol);
    context.move(new_symbol);

    type2tc lentype, sizetype;
    typet olentype = build_type(LENGTH);
    typet osizetype = build_type(SIZE);
    migrate_type(olentype, lentype);
    migrate_type(osizetype, sizetype);

    std::vector<expr2tc> operands;
    operands.push_back(true_expr);
    operands.push_back(constant_int2tc(lentype, l));
    operands.push_back(constant_int2tc(sizetype, l+1));
    constant_struct2tc value(string_struct, operands);

    // initialization
    goto_programt::targett assignment1=initialization.add_instruction(ASSIGN);
    expr2tc new_sym2;
    migrate_expr(new_sym, new_sym2);
    assignment1->code = code_assign2tc(new_sym2, value);
  }

  return symbol2tc(string_struct, identifier);
}

/*******************************************************************\

Function: string_abstractiont::is_zero_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

expr2tc string_abstractiont::is_zero_string(
  const expr2tc &object,
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

expr2tc string_abstractiont::zero_string_length(
  const expr2tc &object,
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

expr2tc string_abstractiont::buffer_size(
  const expr2tc &object,
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

void string_abstractiont::move_lhs_arithmetic(expr2tc &lhs, expr2tc &rhs)
{
  if (is_sub2t(lhs))
  {
    // move op1 to rhs
    expr2tc rest = to_sub2t(lhs).side_1;
    add2tc sum(lhs->type, rhs, to_sub2t(lhs).side_2);
    // overwrite
    rhs = sum;
    lhs = rest;
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
  code_assign2t &assign = to_code_assign2t(target->code);

  if (has_string_macros(assign.target))
  {
    replace_string_macros(assign.target, true, target->location);
    move_lhs_arithmetic(assign.target, assign.source);
  }

  if (has_string_macros(assign.source))
    replace_string_macros(assign.source, false, target->location);

  if (is_pointer_type(assign.target))
    abstract_pointer_assign(dest, target);
  else if (is_char_type(assign.target))
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
  code_assign2t &assign = to_code_assign2t(target->code);

  expr2tc &lhs = assign.target;
  expr2tc rhs = assign.source;

  expr2tc rhsp = assign.source;

  while (is_typecast2t(rhsp))
    rhsp = to_typecast2t(rhsp).from;
  
  // we only care about char pointers for now
  if (!is_char_type(to_pointer_type(rhsp->type).subtype))
    return;

  // assign length and is_zero as well

  goto_programt tmp;

  goto_programt::targett assignment=tmp.add_instruction(ASSIGN);
  assignment->code = code_assign2tc(build(lhs, true), build(rhs, false));
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
  code_assign2t &assign = to_code_assign2t(target->code);

  expr2tc &lhs = assign.target;
  expr2tc &rhs = assign.source;

  // we only care if the constant zero is assigned
  if (!is_constant_int2t(rhs) ||
      !to_constant_int2t(rhs).constant_value.is_zero())
    return;

  if (is_index2t(lhs))
  {
    index2t &idx = to_index2t(lhs);
    goto_programt tmp;

    const expr2tc symbol_buffer = build_symbol_buffer(idx.source_value);

    const expr2tc i1 = member(symbol_buffer, IS_ZERO);
    if (!is_nil_expr(i1))
    {
      goto_programt::targett assignment1=tmp.add_instruction(ASSIGN);
      assignment1->code = code_assign2tc(i1, true_expr);
      assignment1->location=target->location;
      assignment1->local_variables=target->local_variables;
    }

    const expr2tc i2 = member(symbol_buffer, LENGTH);
    if (!is_nil_expr(i2))
    {
      expr2tc new_length = idx.index;
      make_type(new_length, i2->type);

      lessthan2tc cond(new_length, i2);
      if2tc min_expr(i2->type, cond, new_length, i2);

      goto_programt::targett assignment2=tmp.add_instruction(ASSIGN);
      assignment2->code = code_assign2tc(i2, min_expr);
      assignment2->location=target->location;
      assignment2->local_variables=target->local_variables;

      code_assign2t &assign = to_code_assign2t(assignment2->code);
      move_lhs_arithmetic(assign.target, assign.source);
    }

    target++;
    dest.destructive_insert(target, tmp);
  }
}

/*******************************************************************\

Function: string_abstractiont::abstract_function_call

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_abstractiont::abstract_function_call(
  goto_programt &dest __attribute__((unused)),
  goto_programt::targett target)
{
  std::vector<expr2tc> new_args;

  code_function_call2t &call = to_code_function_call2t(target->code);
  const std::vector<expr2tc> &arguments = call.operands;
  
  // Can't cope with non symbols.
  if (!is_symbol2t(call.function))
    return;

  symbolst::const_iterator f_it = 
    context.symbols.find(to_symbol2t(call.function).get_symbol_name());
  if(f_it==context.symbols.end())
    // XXXjmorse - handle function pointer strabs at symex time?
    return;

  // Don't attempt to strabs an absent function.
  if (f_it->second.value.is_nil())
    return;
  
  const code_typet &old_fnc_type = 
    static_cast<const code_typet &>(f_it->second.type);
  type2tc func_type_c;
  migrate_type(old_fnc_type, func_type_c);
  const code_type2t &func_type = to_code_type(func_type_c);
  const std::vector<type2tc> &argument_types = func_type.arguments;

  std::vector<type2tc>::const_iterator arg = argument_types.begin();
  for (std::vector<expr2tc>::const_iterator it1 = arguments.begin();
      it1 != arguments.end(); it1++) {
    const expr2tc actual(*it1);

    new_args.push_back(actual);

    // XXX jmorse migration; see here in the past, arg was being implicity
    // casted from type to expr. Hacking around this could have led to a change
    // in behaviour.
    type2tc tcfree = *arg;
#if 0
    while (is_typecast2t(tcfree))
      tcfree = to_typecast2t(tcfree).from;
#endif
    
    if (is_pointer_type(tcfree) &&
        is_char_type(to_pointer_type(tcfree).subtype))
    {
      if (is_pointer_type(actual))
        new_args.push_back(build(actual, false));
      else
        new_args.push_back(address_of2tc(to_pointer_type(tcfree).subtype,
                                         (build(actual, false))));
    }

    arg++;
    // Don't continue through var-args
    if (arg == argument_types.end())
      break;

    // Uuugh. Arg we're pointing at may (or may not) now be a string struct ptr.
    // Ultimately the fix to this horror is not rewriting program code and
    // signature in the same pass.
    if (is_pointer_type(*arg) && to_pointer_type(*arg).subtype == string_struct)
      arg++;

    if (arg == argument_types.end())
      break;
  }

  // If we have a char return type, receive a returned string struct by passing
  // a string struct pointer as the last argument.
  type2tc fnc_ret_type = func_type.ret_type;
  if (is_pointer_type(fnc_ret_type) &&
      is_char_type(to_pointer_type(fnc_ret_type).subtype)) {
    if (is_nil_expr(call.ret)) {
      type2tc null_type = type2tc(new pointer_type2t(type2tc(new pointer_type2t(string_struct))));
      symbol2tc null(null_type, "NULL");
      new_args.push_back(null);
    } else {
      //XXX jmorse migration guessing; void ptr?
      type2tc ret_type = type2tc(new pointer_type2t(get_empty_type()));
      new_args.push_back(address_of2tc(ret_type, build(call.ret, false)));
    }
  }

  // XXX - previously had a test to ensure that we have the same number of
  // arguments as the function being called. However as we're now changing
  // that number, and we can't guarentee the order these functions are processed
  // in, it's not inpractical.

  call.operands = new_args;
}
