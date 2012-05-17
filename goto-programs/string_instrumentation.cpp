/*******************************************************************\

Module: String Abstraction

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <std_expr.h>
#include <std_code.h>
#include <expr_util.h>
#include <message_stream.h>
#include <arith_tools.h>
#include <config.h>
#include <bitvector.h>

#include <goto-programs/format_strings.h>
#include <ansi-c/c_types.h>

#include "string_instrumentation.h"

/*******************************************************************\

   Class: string_instrumentationt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

class string_instrumentationt:public message_streamt
{
public:
  string_instrumentationt(
    contextt &_context,
    message_handlert &_message_handler):
    message_streamt(_message_handler),
    context(_context),
    ns(_context)
  {
  }

  void operator()(goto_programt &dest);
  void operator()(goto_functionst &dest);

  expr2tc is_zero_string(
    const expr2tc &what,
    bool write=false)
  {
    expr2tc result = expr2tc(new zero_string2t(what));
    return result;
#warning XXXjmorse, string copy write?
    //result.lhs(write);
  }

  expr2tc zero_string_length(
    const expr2tc &what,
    bool write=false)
  {
    expr2tc result = expr2tc(new zero_length_string2t(what));
    return result;
#warning XXXjmorse, string copy write?
    //result.lhs(write);
  }

#if 0
  expr2tc buffer_size(const expr2tc &what)
  {
    exprt result("buffer_size", uint_type());
    result.copy_to_operands(what);
    return result;
  }
#endif

protected:
  contextt &context;
  namespacet ns;

  void make_type(expr2tc &dest, const type2tc &type)
  {
    if (dest->type != type)
      dest = expr2tc(new typecast2t(type, dest));
  }

  void instrument(goto_programt &dest, goto_programt::targett it);
  void do_function_call(goto_programt &dest, goto_programt::targett it);

  // strings
  void do_sprintf (goto_programt &dest, goto_programt::targett it, code_function_call2t &call);
  void do_snprintf(goto_programt &dest, goto_programt::targett it, code_function_call2t &call);
  void do_strcat  (goto_programt &dest, goto_programt::targett it, code_function_call2t &call);
  void do_strncmp (goto_programt &dest, goto_programt::targett it, code_function_call2t &call);
  void do_strchr  (goto_programt &dest, goto_programt::targett it, code_function_call2t &call);
  void do_strrchr (goto_programt &dest, goto_programt::targett it, code_function_call2t &call);
  void do_strstr  (goto_programt &dest, goto_programt::targett it, code_function_call2t &call);
  void do_strtok  (goto_programt &dest, goto_programt::targett it, code_function_call2t &call);
  void do_strerror(goto_programt &dest, goto_programt::targett it, code_function_call2t &call);
  void do_fscanf  (goto_programt &dest, goto_programt::targett it, code_function_call2t &call);

  void do_format_string_read(
    goto_programt &dest,
    goto_programt::const_targett target,
    const std::vector<expr2tc> &arguments,
    unsigned format_string_inx,
    unsigned argument_start_inx,
    const std::string &function_name);

  void do_format_string_write(
    goto_programt &dest,
    goto_programt::const_targett target,
    const std::vector<expr2tc> &arguments,
    unsigned format_string_inx,
    unsigned argument_start_inx,
    const std::string &function_name);

  bool is_string_type(const typet &t) const
  {
    return ((t.id()=="pointer" || t.is_array()) &&
            (t.subtype().id()=="signedbv" || t.subtype().id()=="unsignedbv") &&
            (bv_width(t.subtype())==config.ansi_c.char_width));
  }

  void invalidate_buffer(
    goto_programt &dest,
    goto_programt::const_targett target,
    const expr2tc &buffer,
    const type2tc &buf_type,
    const mp_integer &limit);
};

/*******************************************************************\

Function: string_instrumentation

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentation(
  contextt &context,
  message_handlert &message_handler,
  goto_programt &dest)
{
  string_instrumentationt string_instrumentation(context, message_handler);
  string_instrumentation(dest);
}

/*******************************************************************\

Function: string_instrumentation

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentation(
  contextt &context,
  message_handlert &message_handler,
  goto_functionst &dest)
{
  string_instrumentationt string_instrumentation(context, message_handler);
  string_instrumentation(dest);
}

/*******************************************************************\

Function: string_instrumentationt::operator()

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::operator()(goto_functionst &dest)
{
  for(goto_functionst::function_mapt::iterator
      it=dest.function_map.begin();
      it!=dest.function_map.end();
      it++)
  {
    (*this)(it->second.body);
  }
}

/*******************************************************************\

Function: string_instrumentationt::operator()

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::operator()(goto_programt &dest)
{
  Forall_goto_program_instructions(it, dest)
    instrument(dest, it);
}

/*******************************************************************\

Function: string_instrumentationt::instrument

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::instrument(
  goto_programt &dest,
  goto_programt::targett it)
{
  switch(it->type)
  {
  case ASSIGN:
    break;

  case FUNCTION_CALL:
    do_function_call(dest, it);
    break;

  default:;
  }
}

/*******************************************************************\

Function: string_instrumentationt::do_function_call

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_function_call(
  goto_programt &dest,
  goto_programt::targett target)
{
  code_function_call2t &call = to_code_function_call2t(target->code);
  expr2tc &function = call.function;
  //const exprt &lhs=call.lhs();

  if (is_symbol2t(function))
  {
    const irep_idt &identifier = to_symbol2t(function).name;

    if (identifier=="c::strcoll")
    {
    }
    else if (identifier=="c::strncmp")
      do_strncmp(dest, target, call);
    else if (identifier=="c::strxfrm")
    {
    }
    else if (identifier=="c::strchr")
      do_strchr(dest, target, call);
    else if (identifier=="c::strcspn")
    {
    }
    else if (identifier=="c::strpbrk")
    {
    }
    else if (identifier=="c::strrchr")
      do_strrchr(dest, target, call);
    else if (identifier=="c::strspn")
    {
    }
    else if (identifier=="c::strerror")
      do_strerror(dest, target, call);
    else if (identifier=="c::strstr")
      do_strstr(dest, target, call);
    else if (identifier=="c::strtok")
      do_strtok(dest, target, call);
    else if (identifier=="c::sprintf")
      do_sprintf(dest, target, call);
    else if (identifier=="c::snprintf")
      do_snprintf(dest, target, call);
    else if (identifier=="c::fscanf")
      do_fscanf(dest, target, call);

    dest.compute_targets();
    dest.number_targets();
  }
}

/*******************************************************************\

Function: string_instrumentationt::do_sprintf

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_sprintf(
  goto_programt &dest,
  goto_programt::targett target,
  code_function_call2t &call)
{
  const std::vector<expr2tc> &arguments = call.operands;

  if(arguments.size()<2)
  {
    err_location(target->location);
    throw "sprintf expected to have two or more arguments";
  }

  goto_programt tmp;

  goto_programt::targett assertion=tmp.add_instruction();
  assertion->location=target->location;
  assertion->location.property("string");
  assertion->location.comment("sprintf buffer overflow");
  assertion->local_variables=target->local_variables;

  // in the abstract model, we have to report a
  // (possibly false) positive here
  assertion->make_assertion(expr2tc(new constant_bool2t(false)));

  do_format_string_read(tmp, target, arguments, 1, 2, "sprintf");

  if (!is_nil_expr(call.ret)) {
    goto_programt::targett return_assignment=tmp.add_instruction(ASSIGN);
    return_assignment->location=target->location;
    return_assignment->local_variables=target->local_variables;

    expr2tc rhs = expr2tc(new sideeffect2t(call.ret->type, expr2tc(),
                                           expr2tc(), type2tc(),
                                           sideeffect2t::nondet));

    return_assignment->code = expr2tc(new code_assign2t(call.ret, rhs));
  }

  target->make_skip();
  dest.insert_swap(target, tmp);
}

/*******************************************************************\

Function: string_instrumentationt::do_snprintf

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_snprintf(
  goto_programt &dest,
  goto_programt::targett target,
  code_function_call2t &call)
{
  const std::vector<expr2tc> &arguments = call.operands;

  std::cerr << "Warning, snprintf not string-abstracted post irep migration" << std::endl;
  abort();

#warning XXXjmorse, killed snprintf string abstraction due to buffer_side irep. It wouldn't have worked previously anyway.
#if 0
  if(arguments.size()<3)
  {
    err_location(target->location);
    throw "snprintf expected to have three or more arguments";
  }

  goto_programt tmp;

  goto_programt::targett assertion=tmp.add_instruction();
  assertion->location=target->location;
  assertion->location.property("string");
  assertion->location.comment("snprintf buffer overflow");
  assertion->local_variables=target->local_variables;

  exprt bufsize = buffer_size(arguments[0]);
  assertion->make_assertion(binary_relation_exprt(bufsize, ">=", arguments[1]));

  do_format_string_read(tmp, target, arguments, 2, 3, "snprintf");

  if(call.lhs().is_not_nil())
  {
    goto_programt::targett return_assignment=tmp.add_instruction(ASSIGN);
    return_assignment->location=target->location;
    return_assignment->local_variables=target->local_variables;

    exprt rhs=side_effect_expr_nondett(call.lhs().type());
    rhs.location()=target->location;

    return_assignment->code=code_assignt(call.lhs(), rhs);
  }

  target->make_skip();
  dest.insert_swap(target, tmp);
#endif
}

/*******************************************************************\

Function: string_instrumentationt::do_fscanf

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_fscanf(
  goto_programt &dest,
  goto_programt::targett target,
  code_function_call2t &call)
{
  const std::vector<expr2tc> &arguments = call.operands;

  if(arguments.size()<2)
  {
    err_location(target->location);
    throw "fscanf expected to have two or more arguments";
  }

  goto_programt tmp;

  do_format_string_write(tmp, target, arguments, 1, 2, "fscanf");

  if (!is_nil_expr(call.ret))
  {
    goto_programt::targett return_assignment=tmp.add_instruction(ASSIGN);
    return_assignment->location=target->location;
    return_assignment->local_variables=target->local_variables;

    expr2tc rhs = expr2tc(new sideeffect2t(call.ret->type, expr2tc(), expr2tc(),
                                           type2tc(), sideeffect2t::nondet));

    return_assignment->code = expr2tc(new code_assign2t(call.ret, rhs));
  }

  target->make_skip();
  dest.insert_swap(target, tmp);
}

/*******************************************************************\

Function: string_instrumentationt::do_format_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_format_string_read(
  goto_programt &dest,
  goto_programt::const_targett target,
  const std::vector<expr2tc> &arguments,
  unsigned format_string_inx,
  unsigned argument_start_inx,
  const std::string &function_name)
{
  const expr2tc &format_arg = arguments[format_string_inx];

  std::cerr << "do_format_string_read: is a victim of irep migration" << std::endl;
  abort();
#if 0
  if (is_address_of2t(format_arg)) {
    const address_of2t &addrof = to_address_of2t(format_art);
    if (is_index2t(addrof.source_value)) {
      const index2t &idx = to_index2t(addrof.source_value);
      if (is_constant_string2t(idx.source_value)) {
        format_token_listt token_list;
        parse_format_string(migrate_expr_back(idx.source_value), token_list);

        unsigned args=0;

        for(format_token_listt::const_iterator it=token_list.begin();
            it!=token_list.end();
            it++)
        {
          if(it->type==format_tokent::STRING)
          {
            const expr2tc &arg = arguments[argument_start_inx+args];
            const type2tc &arg_type = arg->type;

            if (is_constant_string2t(arg)) // we don't need to check constants
            {
              goto_programt::targett assertion=dest.add_instruction();
              assertion->location=target->location;
              assertion->location.property("string");
              std::string comment("zero-termination of string argument of ");
              comment += function_name;
              assertion->location.comment(comment);
              assertion->local_variables=target->local_variables;

              expr2tc temp(arg);

              if (!is_pointer_type(arg_type))
              {
                const array_type2t &arr_ref = to_array_type(arg_type);

                expr2tc zero =
                  expr2tc(new constant_int2t(uint_type2(), BigInt(0)));
                expr2tc index = expr2tc(new index2t(arr_ref.subtype,temp,zero));
                temp = expr2tc(new address_of2t(index->type, index));
              }

              assertion->make_assertion(is_zero_string(temp));
            }
          }

          if(it->type!=format_tokent::TEXT &&
             it->type!=format_tokent::UNKNOWN) args++;

          if(find(it->flags.begin(), it->flags.end(), format_tokent::ASTERISK)!=
             it->flags.end())
            args++; // just eat the additional argument
        }
      }
      else // non-const format string
  {
    goto_programt::targett format_ass=dest.add_instruction();
    format_ass->make_assertion(is_zero_string(arguments[1]));
    format_ass->location=target->location;
    format_ass->location.property("string");
    std::string comment("zero-termination of format string of ");
    comment += function_name;
    format_ass->location.comment(comment);
    format_ass->local_variables=target->local_variables;

    for(unsigned i=2; i<arguments.size(); i++)
    {
      const expr2tc &arg = arguments[i];
      const type2tc &arg_type = arguments[i]->type;

      if (!is_constant_string2t(arguments[i]) && is_string_type(arg_type))
      {
        goto_programt::targett assertion=dest.add_instruction();
        assertion->location=target->location;
        assertion->location.property("string");
        std::string comment("zero-termination of string argument of ");
        comment += function_name;
        assertion->location.comment(comment);
        assertion->local_variables=target->local_variables;

        expr2tc temp(arg);

        if (!is_pointer_type(arg_type))
        {
          const array_type2t &arr_ref = to_array_type(arg_type);

          expr2tc zero =
            expr2tc(new constant_int2t(uint_type2(), BigInt(0)));
          expr2tc index = expr2tc(new index2t(arr_ref.subtype,temp,zero));
          temp = expr2tc(new address_of2t(index->type, index));
        }

        assertion->make_assertion(is_zero_string(temp));
      }
    }
  }
#endif
}

/*******************************************************************\

Function: string_instrumentationt::do_format_string_write

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_format_string_write(
  goto_programt &dest,
  goto_programt::const_targett target,
  const std::vector<expr2tc> &arguments,
  unsigned format_string_inx,
  unsigned argument_start_inx,
  const std::string &function_name)
{
  const expr2tc &format_arg = arguments[format_string_inx];

  std::cerr << "do_format_string_write: is a victim of irep migration" << std::endl;
  abort();

#if 0
  if (is_address_of2t(format_arg)) {
    const address_of2t &addrof = to_address_of2t(format_arg);
    if (is_index2t(addrof.source_value)) {
      const index2t &index = to_index2t(addrof.source_value);
      if (is_string_constant2t(index.source_value)) {
        format_token_listt token_list;
        parse_format_string(migrate_expr_back(index.source_value), token_list);

        unsigned args=0;

        for(format_token_listt::const_iterator it=token_list.begin();
            it!=token_list.end();
            it++)
        {
          if(find(it->flags.begin(), it->flags.end(), format_tokent::ASTERISK)!=
             it->flags.end())
            continue; // asterisk means `ignore this'

          switch(it->type)
          {
            case format_tokent::STRING:
            {

              const expr2tc &argument = arguments[argument_start_inx+args];
              const type2tc &arg_type = argument->type;

              goto_programt::targett assertion=dest.add_instruction();
              assertion->location=target->location;
              assertion->location.property("string");
              std::string comment("format string buffer overflow in ");
              comment += function_name;
              assertion->location.comment(comment);
              assertion->local_variables=target->local_variables;

              if(it->field_width!=0)
              {
                expr2tc fwidth =
                  expr2tc(new constant_int2t(uint_type2(), it->field_width));
                expr2tc one =
                  expr2tc(new constant_int2t(uint_type2(), Bigint(1)));
                expr2tc(new add2t(uint_type2(), fwidth, one));

                exprt fw_lt_bs;

                if (is_pointer_type(arg_type))
                  fw_lt_bs = expr2tc(new lessthanequal2t(fw_1, buffer_size aaaarrrrrgghhh my eyes!
                  fw_lt_bs=binary_relation_exprt(fw_1, "<=", buffer_size(argument));
                else
                {
                  index_exprt index;
                  index.array()=argument;
                  index.index()=gen_zero(uint_type());
                  address_of_exprt aof(index);
                  fw_lt_bs=binary_relation_exprt(fw_1, "<=", buffer_size(aof));
                }

                assertion->make_assertion(fw_lt_bs);
              }
              else
              {
                // this is a possible overflow.
                assertion->make_assertion(false_exprt());
              }

              // now kill the contents
              invalidate_buffer(dest, target, argument, arg_type, it->field_width);

              args++;
              break;
            }
            case format_tokent::TEXT:
            case format_tokent::UNKNOWN:
            {
              // nothing
              break;
            }
            default: // everything else
            {
              const exprt &argument=arguments[argument_start_inx+args];
              const typet &arg_type=ns.follow(argument.type());

              goto_programt::targett assignment=dest.add_instruction(ASSIGN);
              assignment->location=target->location;
              assignment->local_variables=target->local_variables;

              exprt lhs("dereference", arg_type.subtype());
              lhs.copy_to_operands(argument);

              exprt rhs=side_effect_expr_nondett(lhs.type());
              rhs.location()=target->location;

              assignment->code=code_assignt(lhs, rhs);

              args++;
              break;
            }
          }
        }
      }
  else // non-const format string
  {
    for(unsigned i=argument_start_inx; i<arguments.size(); i++)
    {
      const typet &arg_type=ns.follow(arguments[i].type());

      // Note: is_string_type() is a `good guess' here. Actually
      // any of the pointers could point into an array. But it
      // would suck if we had to invalidate all variables.
      // Luckily this case isn't needed too often.
      if(is_string_type(arg_type))
      {
        goto_programt::targett assertion=dest.add_instruction();
        assertion->location=target->location;
        assertion->location.property("string");
        std::string comment("format string buffer overflow in ");
        comment += function_name;
        assertion->location.comment(comment);
        assertion->local_variables=target->local_variables;
        // as we don't know any field width for the %s that
        // should be here during runtime, we just report a
        // possibly false positive
        assertion->make_assertion(false_exprt());

        invalidate_buffer(dest, target, arguments[i], arg_type, 0);
      }
      else
      {
        goto_programt::targett assignment = dest.add_instruction(ASSIGN);
        assignment->location=target->location;
        assignment->local_variables=target->local_variables;

        exprt lhs("dereference", arg_type.subtype());
        lhs.copy_to_operands(arguments[i]);

        exprt rhs=side_effect_expr_nondett(lhs.type());
        rhs.location()=target->location;

        assignment->code=code_assignt(lhs, rhs);
      }
    }
  }
#endif
}

/*******************************************************************\

Function: string_instrumentationt::do_strncmp

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_strncmp(
  goto_programt &dest,
  goto_programt::targett target,
  code_function_call2t &call)
{
}

/*******************************************************************\

Function: string_instrumentationt::do_strchr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_strchr(
  goto_programt &dest,
  goto_programt::targett target,
  code_function_call2t &call)
{
  const std::vector<expr2tc> &arguments = call.operands;

  if(arguments.size()!=2)
  {
    err_location(target->location);
    throw "strchr expected to have two arguments";
  }

  goto_programt tmp;

  goto_programt::targett assertion=tmp.add_instruction();
  assertion->make_assertion(is_zero_string(arguments[0]));
  assertion->location=target->location;
  assertion->location.property("string");
  assertion->location.comment("zero-termination of string argument of strchr");
  assertion->local_variables=target->local_variables;

  target->make_skip();
  dest.insert_swap(target, tmp);
}

/*******************************************************************\

Function: string_instrumentationt::do_strrchr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_strrchr(
  goto_programt &dest,
  goto_programt::targett target,
  code_function_call2t &call)
{
  const std::vector<expr2tc> &arguments = call.operands;

  if(arguments.size()!=2)
  {
    err_location(target->location);
    throw "strrchr expected to have two arguments";
  }

  goto_programt tmp;

  goto_programt::targett assertion=tmp.add_instruction();
  assertion->make_assertion(is_zero_string(arguments[0]));
  assertion->location=target->location;
  assertion->location.property("string");
  assertion->location.comment("zero-termination of string argument of strrchr");
  assertion->local_variables=target->local_variables;

  target->make_skip();
  dest.insert_swap(target, tmp);
}

/*******************************************************************\

Function: string_instrumentationt::do_strstr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_strstr(
  goto_programt &dest,
  goto_programt::targett target,
  code_function_call2t &call)
{
  const std::vector<expr2tc> &arguments = call.operands;

  if(arguments.size()!=2)
  {
    err_location(target->location);
    throw "strstr expected to have two arguments";
  }

  goto_programt tmp;

  goto_programt::targett assertion0=tmp.add_instruction();
  assertion0->make_assertion(is_zero_string(arguments[0]));
  assertion0->location=target->location;
  assertion0->location.property("string");
  assertion0->location.comment("zero-termination of 1st string argument of strstr");
  assertion0->local_variables=target->local_variables;

  goto_programt::targett assertion1=tmp.add_instruction();
  assertion1->make_assertion(is_zero_string(arguments[1]));
  assertion1->location=target->location;
  assertion1->location.property("string");
  assertion1->location.comment("zero-termination of 2nd string argument of strstr");
  assertion1->local_variables=target->local_variables;

  target->make_skip();
  dest.insert_swap(target, tmp);
}

/*******************************************************************\

Function: string_instrumentationt::do_strtok

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_strtok(
  goto_programt &dest,
  goto_programt::targett target,
  code_function_call2t &call)
{
  const std::vector<expr2tc> &arguments = call.operands;

  if(arguments.size()!=2)
  {
    err_location(target->location);
    throw "strtok expected to have two arguments";
  }

  goto_programt tmp;

  goto_programt::targett assertion0=tmp.add_instruction();
  assertion0->make_assertion(is_zero_string(arguments[0]));
  assertion0->location=target->location;
  assertion0->location.property("string");
  assertion0->location.comment("zero-termination of 1st string argument of strtok");
  assertion0->local_variables=target->local_variables;

  goto_programt::targett assertion1=tmp.add_instruction();
  assertion1->make_assertion(is_zero_string(arguments[1]));
  assertion1->location=target->location;
  assertion1->location.property("string");
  assertion1->location.comment("zero-termination of 2nd string argument of strtok");
  assertion1->local_variables=target->local_variables;

  target->make_skip();
  dest.insert_swap(target, tmp);
}

/*******************************************************************\

Function: string_instrumentationt::do_strerror

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::do_strerror(
  goto_programt &dest,
  goto_programt::targett it,
  code_function_call2t &call)
{
  if (is_nil_expr(call.ret))
  {
    it->make_skip();
    return;
  }

  irep_idt identifier_buf="c::__strerror_buffer";
  irep_idt identifier_size="c::__strerror_buffer_size";

  if(context.symbols.find(identifier_buf)==context.symbols.end())
  {
    symbolt new_symbol_size;
    new_symbol_size.base_name="__strerror_buffer_size";
    new_symbol_size.pretty_name=new_symbol_size.base_name;
    new_symbol_size.name=identifier_size;
    new_symbol_size.mode="C";
    new_symbol_size.type=uint_type();
    new_symbol_size.is_statevar=true;
    new_symbol_size.lvalue=true;
    new_symbol_size.static_lifetime=true;

    array_typet type;
    type.subtype()=char_type();
    type.size()=symbol_expr(new_symbol_size);
    symbolt new_symbol_buf;
    new_symbol_buf.mode="C";
    new_symbol_buf.type=type;
    new_symbol_buf.is_statevar=true;
    new_symbol_buf.lvalue=true;
    new_symbol_buf.static_lifetime=true;
    new_symbol_buf.base_name="__strerror_buffer";
    new_symbol_buf.pretty_name=new_symbol_buf.base_name;
    new_symbol_buf.name="c::"+id2string(new_symbol_buf.base_name);

    context.move(new_symbol_buf);
    context.move(new_symbol_size);
  }

  const symbolt &symbol_size=ns.lookup(identifier_size);
  const symbolt &symbol_buf=ns.lookup(identifier_buf);

  goto_programt tmp;

  {
    goto_programt::targett assignment1=tmp.add_instruction(ASSIGN);
    expr2tc nondet_size = expr2tc(new sideeffect2t(uint_type2(), expr2tc(),
                                                   expr2tc(), type2tc(),
                                                   sideeffect2t::nondet));

    exprt sym = symbol_expr(symbol_size);
    expr2tc new_sym;
    migrate_expr(sym, new_sym);
    expr2tc code = expr2tc(new code_assign2t(new_sym, nondet_size));
    assignment1->code = code;
    assignment1->location=it->location;
    assignment1->local_variables=it->local_variables;

    goto_programt::targett assumption1=tmp.add_instruction();

    exprt sym_expr = symbol_expr(symbol_size);
    exprt zero = gen_zero(symbol_size.type);
    expr2tc new_expr;
    migrate_expr(sym_expr, new_expr);
    expr2tc new_zero;
    migrate_expr(zero, new_zero);
    expr2tc noneq = expr2tc(new notequal2t(new_expr, new_zero));

    assumption1->make_assumption(noneq);
    assumption1->location=it->location;
    assumption1->local_variables=it->local_variables;
  }

  // return a pointer to some magic buffer
  expr2tc new_sym, new_zero;
  exprt sym = symbol_expr(symbol_buf);
  migrate_expr(sym, new_sym);
  exprt zero = gen_zero(uint_type());
  migrate_expr(zero, new_zero);

  expr2tc index = expr2tc(new index2t(char_type2(), new_sym, new_zero));
  expr2tc ptr = expr2tc(new address_of2t(char_type2(), index));

  // make that zero-terminated
  {
    goto_programt::targett assignment2=tmp.add_instruction(ASSIGN);
    expr2tc zero_string = expr2tc(new zero_string2t(ptr));
    expr2tc true_val = expr2tc(new constant_bool2t(true));
    expr2tc assign = expr2tc(new code_assign2t(zero_string, true_val));

    assignment2->code = assign;
    assignment2->location=it->location;
    assignment2->local_variables=it->local_variables;
  }

  // assign address
  {
    goto_programt::targett assignment3=tmp.add_instruction(ASSIGN);
    expr2tc rhs = ptr;
    make_type(rhs, call.ret->type);

    expr2tc assign = expr2tc(new code_assign2t(call.ret, rhs));
    assignment3->code = assign;
    assignment3->location=it->location;
    assignment3->local_variables=it->local_variables;
  }

  it->make_skip();
  dest.insert_swap(it, tmp);
}

/*******************************************************************\

Function: string_instrumentationt::invalidate_buffer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void string_instrumentationt::invalidate_buffer(
  goto_programt &dest,
  goto_programt::const_targett target,
  const expr2tc &buffer,
  const type2tc &buf_type,
  const mp_integer &limit)
{
  irep_idt cntr_id="string_instrumentation::$counter";

  std::cerr << "XXX jmorse, invalid buffer is a victim of irep migration" << std::endl;
  abort();

#if 0
  if(context.symbols.find(cntr_id)==context.symbols.end())
  {
    symbolt new_symbol;
    new_symbol.base_name="$counter";
    new_symbol.pretty_name=new_symbol.base_name;
    new_symbol.name=cntr_id;
    new_symbol.mode="C";
    new_symbol.type=uint_type();
    new_symbol.is_statevar=true;
    new_symbol.lvalue=true;
    new_symbol.static_lifetime=true;

    context.move(new_symbol);
  }

  const symbolt &cntr_sym=ns.lookup(cntr_id);

  // create a loop that runs over the buffer
  // and invalidates every element

  goto_programt::targett init=dest.add_instruction(ASSIGN);
  init->location=target->location;
  init->local_variables=target->local_variables;

  type2tc cntr_sym_type;
  migrate_expr(cntr_sym.type, cntr_sym_type);
  exprt sym = symbol_expr(cntr_sym);
  expr2tc sym_expr, zero;
  migrate_expr(sym, sym_expr);
  zero = expr2tc(new constant_int2t(cntr_sym_type, BigInt(0)));
  expr2tc assign = expr2tc(new code_assign2t(sym_expr, zero));
  init->code = assign;

  goto_programt::targett check=dest.add_instruction();
  check->location=target->location;
  check->local_variables=target->local_variables;

  goto_programt::targett invalidate=dest.add_instruction(ASSIGN);
  invalidate->location=target->location;
  invalidate->local_variables=target->local_variables;

  goto_programt::targett increment=dest.add_instruction(ASSIGN);
  increment->location=target->location;
  increment->local_variables=target->local_variables;

  exprt cntr_sym_expr = symbol_expr(cntr_sym);
  expr2tc cntr_new_expr, one;
  migrate_expr(cntr_sym_expr, cntr_new_expr);
  one = expr2tc(new constant_int2t(uint_type2(), BigInt(2)));

  expr2tc plus = expr2tc(new add2t(uint_type2(), cntr_new_expr, one));
  assign = expr2tc(new code_assign2t(cntr_new_expr, plus));

  goto_programt::targett back=dest.add_instruction();
  back->location=target->location;
  back->local_variables=target->local_variables;
  back->make_goto(check);
  back->guard = expr2tc(new constant_bool2t(true));

  goto_programt::targett exit=dest.add_instruction();
  exit->location=target->location;
  exit->local_variables=target->local_variables;
  exit->make_skip();

  expr2tc cnt_bs, bufp;

  if (is_pointer_type(buf_type))
    bufp = buffer;
  else
  {
    expr2tc zero = expr2tc(new constant_int2t(uint_type2(), BigInt(0)));
    const array_type2t &arr_type = to_array_type(buf_type);
    expr2tc index = expr2tc(new index2t(arr_type.subtype, buffer, zero));
    bufp = expr2tc(new address_of2t(arr_type.subtype, index));
  }

  exprt sym_expr = symbol_expr(cntr_sym);
  expr2tc new_sym_expr;
  migrate_expr(sym_expr, new_sym_expr);
  expr2tc b_plus_i = expr2tc(new add2t(bufp->type, bufp, new_sym_expr));

  expr2tc deref = expr2tc(new dereference2t("dereference",
                  is_pointer_type(buf_type) ? to_pointer_type(buf_type).subtype
                                            : to_array_type(buf_type).subtype,
                  b_plus_i));



  check->make_goto(exit);

  if(limit==0)
    check->guard = expr2tc(new greaterthanequal2t(new_sym_expr, 
          jhhhngnggg, more buffer size gumph!


          binary_relation_exprt(symbol_expr(cntr_sym), ">=",
                                buffer_size(bufp));
  else
    check->guard=
          binary_relation_exprt(symbol_expr(cntr_sym), ">",
                                from_integer(limit, uint_type()));

  exprt nondet=side_effect_expr_nondett(buf_type.subtype());
  invalidate->code=code_assignt(deref, nondet);
#endif
}
