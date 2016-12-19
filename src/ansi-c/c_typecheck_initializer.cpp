/*******************************************************************\

Module: ANSI-C Conversion / Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <arith_tools.h>
#include <string2array.h>
#include <config.h>
#include <i2string.h>
#include <type_eq.h>
#include <std_types.h>
#include <expr_util.h>
#include <simplify_expr.h>
#include <cprover_prefix.h>
#include <prefix.h>
#include <std_types.h>

#include "c_types.h"
#include "c_typecheck_base.h"

/*******************************************************************\

Function: c_typecheck_baset::zero_initializer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool c_typecheck_baset::zero_initializer(
  exprt &value,
  const typet &type) const
{
  const std::string &type_id=type.id_string();

  if(type_id=="bool")
  {
    value.make_false();
    return false;
  }
  else if(type_id=="unsignedbv" ||
          type_id=="signedbv" ||
          type_id=="floatbv" ||
          type_id=="fixedbv" ||
          type_id=="pointer")
  {
    value=gen_zero(type);
    return false;
  }
  else if(type_id=="code")
    return false;
  else if(type_id=="c_enum" ||
          type_id=="incomplete_c_enum")
  {
    value=exprt("constant", type);
    value.value(i2string(0));
    return false;
  }
  else if(type_id=="array")
  {
    const array_typet &array_type=to_array_type(type);

    exprt tmpval;
    if(zero_initializer(tmpval, array_type.subtype())) return true;

    const exprt &size_expr=array_type.size();

    if(size_expr.id()=="infinity")
    {
    }
    else
    {
      mp_integer size;

      if(to_integer(size_expr, size))
        return true;

      // Permit GCC zero sized arrays; disallow negative sized arrays.
      // Cringe slightly when doing it though.
      if (size < 0) return true;
    }

    value=exprt("array_of", type);
    value.move_to_operands(tmpval);

    return false;
  }
  else if(type_id=="struct")
  {
    const irept::subt &components=
      type.components().get_sub();

    value=exprt("struct", type);

    forall_irep(it, components)
    {
      exprt tmp;

      if(zero_initializer(tmp, (const typet &)it->type()))
        return true;

      value.move_to_operands(tmp);
    }

    return false;
  }
  else if(type_id=="union")
  {
    const irept::subt &components=
      type.components().get_sub();

    value=exprt("union", type);

    if(components.empty())
      return true;

    value.component_name(components.front().name());

    exprt tmp;

    if(zero_initializer(tmp, (const typet &)components.front().type()))
      return true;

    value.move_to_operands(tmp);

    return false;
  }
  else if(type_id=="symbol")
    return zero_initializer(value, follow(type));

  return true;
}

/*******************************************************************\

Function: c_typecheck_baset::do_initializer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::do_initializer(
  exprt &value,
  typet &type,
  bool force_constant)
{
  if(type.id()=="symbol")
  {
    const irep_idt &identifier=type.identifier();
    symbolt* s = context.find_symbol(identifier);

    if(s == nullptr)
    {
      str << "failed to find symbol `" << identifier << "'";
      throw 0;
    }

    do_initializer(value, s->type, force_constant);
    return;
  }

  value=do_initializer_rec(value, type, force_constant);

  if(type.id()=="incomplete_array")
  {
    assert(value.type().is_array());
    type=value.type();
  }
}

/*******************************************************************\

Function: c_typecheck_baset::do_initializer_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt c_typecheck_baset::do_initializer_rec(
  const exprt &value,
  const typet &type,
  bool force_constant)
{
  const typet &full_type=follow(type);

  if(full_type.id()=="incomplete_struct")
  {
    err_location(value);
    str << "type `"
        << to_string(full_type) << "' is still incomplete -- cannot initialize";
    throw 0;
  }

  if(value.id()=="designated_list")
  {
    // Can't designated-initialize anything but a struct; however an array
    // initializer with no elements can be interpreted as an empty designated
    // list, so permit that.
    if(full_type.id()!="struct" && value.operands().size() != 0)
    {
      err_location(value);
      str << "designated initializers cannot initialize `"
          << to_string(full_type) << "'";
      throw 0;
    }

    if (full_type.id() == "struct")
      return do_designated_initializer(value, to_struct_type(full_type), force_constant);
  }

  if(full_type.id()=="incomplete_array" ||
     full_type.is_array() ||
     full_type.id()=="struct" ||
     full_type.id()=="union")
  {
    if(value.id()=="constant" &&
       follow(value.type()).id()=="incomplete_array")
    {
      init_statet state(value);
      return do_initializer_rec(state, type, force_constant, false);
    }
    else if(value.id()=="string-constant")
    {
      // we only do this for arrays, not for structs
      if(full_type.is_array() ||
         full_type.id()=="incomplete_array")
      {
        exprt tmp;
        string2array(value, tmp);
        init_statet state(tmp);
        return do_initializer_rec(state, type, force_constant, false);
      }
      else
      {
        err_location(value);
        str << "string constants cannot initialize struct types";
        throw 0;
      }
    }
    else if(follow(value.type())==full_type)
    {
      return value;
    }
    else if (value.id() == "designated_list" && value.operands().size() == 0)
    {
      // Zero size array initializer
      exprt tmp;
      init_statet state(tmp);
      return do_initializer_incomplete_array(state, full_type, force_constant);
    }
    else
    {
      err_location(value);
      str << "invalid initializer";
      throw 0;
    }
  }
  else
  {
    if(value.type().id()=="incomplete_array")
      if(value.operands().size()==1)
        return do_initializer_rec(value.op0(), type, force_constant); // other types

    exprt tmp(value);
    implicit_typecast(tmp, type);
    return tmp;
  }
}

/*******************************************************************\

Function: c_typecheck_baset::do_initializer_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt c_typecheck_baset::do_initializer_rec(
  init_statet &state,
  const typet &type,
  bool force_constant,
  bool go_down)
{
  // we may go down one level, but we don't have to
  if(go_down &&
     state.has_next() &&
     state->type().id()=="incomplete_array" &&
     state->id()=="constant")
  {
    init_statet tmp_state(*state);
    state++;

    return do_initializer_rec(tmp_state, type, force_constant, false);
  }

  const typet &full_type=follow(type);

  if(full_type.is_array())
    return do_initializer_array(state, to_array_type(full_type), force_constant);
  else if(full_type.id()=="incomplete_array")
    return do_initializer_incomplete_array(state, full_type, force_constant);
  else if(full_type.id()=="struct")
    return do_initializer_struct(state, to_struct_type(full_type), force_constant);
  else if(full_type.id()=="union")
    return do_initializer_union(state, to_union_type(full_type), force_constant);
  else
  {
    // The initializer for a scalar shall be a single expression,
    // * optionally enclosed in braces. *

    exprt result=*state;
    state++;
    implicit_typecast(result, type);
    return result;
  }
}

/*******************************************************************\

Function: c_typecheck_baset::do_initializer_array

  Inputs:

 Outputs:

  Purpose:

\*******************************************************************/

exprt c_typecheck_baset::do_initializer_array(
  init_statet &state,
  const array_typet &type,
  bool force_constant)
{
  // get size

  mp_integer mp_size;

  if(to_integer(type.size(), mp_size))
  {
    err_location(type);
    str << "array size `" << to_string(type.size())
        << "' is not a constant";
    throw 0;
  }

  if(mp_size<0)
  {
    err_location(type);
    str << "array size `" << to_string(type.size())
        << "' is negative";
    throw 0;
  }

  // magic number
  if(mp_size>1000000)
  {
    err_location(type);
    str << "array size `" << to_string(type.size())
        << "' is probably too large";
    throw 0;
  }

  unsigned long size=mp_size.to_ulong();

  // build array constant
  exprt result("constant", type);

  result.operands().resize(size);

  // grab initializers
  for(unsigned pos=0; pos<size; pos++)
  {
    if(!state.has_next())
    {
      exprt zero;

      const typet &subtype=follow(type.subtype());

      if(zero_initializer(zero, subtype))
      {
        err_location(type);
        str << "failed to initialize type "
            << to_string(subtype) << " for zero padding";
        throw 0;
      }

      for(; pos<size; pos++)
        result.operands()[pos]=zero;

      break;
    }

    exprt &r=result.operands()[pos];

    r=do_initializer_rec(state, type.subtype(), force_constant, true);
  }

  return result;
}

/*******************************************************************\

Function: c_typecheck_baset::do_initializer_incomplete_array

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt c_typecheck_baset::do_initializer_incomplete_array(
  init_statet &state,
  const typet &type,
  bool force_constant)
{
  // build array constant
  exprt result("constant", type);

  // lucky guess
  result.reserve_operands(state.remaining());

  const typet &subtype=follow(follow(type).subtype());

  // grab initializers
  while(state.has_next())
  {
    result.copy_to_operands(
      do_initializer_rec(state, subtype, force_constant, true));
  }

  // get size
  unsigned s=result.operands().size();

  // set size
  result.type().id("array");
  result.type().size(from_integer(s, int_type()));

  return result;
}

/*******************************************************************\

Function: c_typecheck_baset::do_initializer_struct

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt c_typecheck_baset::do_initializer_struct(
  init_statet &state,
  const struct_typet &type,
  bool force_constant)
{
  if(state->id() == "designated_list")
  {
    exprt e=do_designated_initializer(*state, type, force_constant);
    state++;
    return e;
  }

  const struct_typet::componentst &components=
    type.components();

  exprt result("struct", type);

  result.reserve_operands(components.size());

  for(struct_typet::componentst::const_iterator
      it=components.begin();
      it!=components.end();
      it++)
  {
    const typet &op_type=it->type();

    if(state.has_next())
    {
      result.copy_to_operands(
        do_initializer_rec(state, op_type, force_constant, true));
    }
    else
    {
      exprt zero;

      if(zero_initializer(zero, op_type))
      {
        err_location(type);
        str << "failed to initialize type "
            << to_string(op_type) << " for struct padding";
        throw 0;
      }

      result.move_to_operands(zero);
    }
  }

  return result;
}

/*******************************************************************\

Function: c_typecheck_baset::do_initializer_union

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt c_typecheck_baset::do_initializer_union(
  init_statet &state,
  const union_typet &type,
  bool force_constant)
{

  if(state->id() == "designated_list")
  {
    exprt e=do_designated_union_initializer(*state, type, force_constant);
    state++;
    return e;
  }

  if(!state.has_next())
  {
    exprt zero;
    zero_initializer(zero, type);
    return zero;
  }

  const union_typet::componentst &components=
    type.components();

  exprt result("union", type);

  if(components.empty())
  {
    err_location(*state);
    str << "initialization of empty union";
    throw 0;
  }

  const typet &op_type=components.front().type();
  result.copy_to_operands(
    do_initializer_rec(state, op_type, force_constant, true));
  result.component_name(components.front().name());

  return result;
}

/*******************************************************************\

Function: c_typecheck_baset::do_initializer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::do_initializer(symbolt &symbol)
{
  // this one doesn't need initialization
  if(has_prefix(id2string(symbol.name), CPROVER_PREFIX "constant_infinity"))
    return;

  if(symbol.static_lifetime)
  {
    if(symbol.value.is_nil())
    {
      const typet &final_type=follow(symbol.type);

      if(final_type.id()!="incomplete_struct" &&
         final_type.id()!="incomplete_array" &&
         !symbol.is_extern) // Don't zero-init externs
      {
        // zero initializer
        if(zero_initializer(symbol.value, symbol.type))
        {
          err_location(symbol.location);
          str << "failed to zero-initialize symbol `"
              << symbol.display_name() << "' with type `"
              << to_string(symbol.type) << "'";
          throw 0;
        }

        symbol.value.zero_initializer(true);
      }
    }
    else
    {
      typecheck_expr(symbol.value);
      do_initializer(symbol.value, symbol.type, true);
    }
  }
  else if(!symbol.is_type)
  {
    const typet &final_type=follow(symbol.type);

    if(final_type.id()=="incomplete_c_enum" ||
       final_type.id()=="c_enum")
    {
      if(symbol.is_macro)
      {
        // these must have a constant value
        assert(symbol.value.is_not_nil());
        typecheck_expr(symbol.value);
        locationt location=symbol.value.location();
        do_initializer(symbol.value, symbol.type, true);
        make_constant(symbol.value);
      }
    }
  }
}

/*******************************************************************\

Function: c_typecheck_baset::do_designated_initializer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt c_typecheck_baset::do_designated_initializer(
  const exprt &value,
  const struct_typet &struct_type,
  bool force_constant)
{
  assert(value.id()=="designated_list");

  exprt result("struct", struct_type);

  const struct_typet::componentst &components=
    struct_type.components();

  // start with NIL
  result.operands().resize(
    components.size(),
    static_cast<const exprt &>(get_nil_irep()));

  forall_operands(it, value)
  {
    const exprt &initializer=*it;

    assert(initializer.operands().size()==1);

    const irep_idt &component_name=initializer.component_name();

    if(!struct_type.has_component(component_name))
    {
      err_location(initializer);
      str << "failed to find component `" << component_name << "'";
      throw 0;
    }

    unsigned number=struct_type.component_number(component_name);

    assert(number<result.operands().size());

    result.operands()[number]=do_initializer_rec(
      initializer.op0(),
      components[number].type(),
      force_constant);
  }

  // NIL left? zero initialize!
  for(unsigned i=0; i<result.operands().size(); i++)
  {
    exprt &initializer=result.operands()[i];

    if(initializer.is_nil())
      zero_initializer(initializer, components[i].type());
  }

  return result;
}

exprt c_typecheck_baset::do_designated_union_initializer(
  const exprt &value,
  const union_typet &union_type,
  bool force_constant)
{
  assert(value.id()=="designated_list");
  assert(value.operands().size() == 1);

  exprt result("union", union_type);

  // We don't in fact have to lay out a series of fields. Because this is a
  // union, all we do to represent a constant set operand 0 to something of the
  // type of one of the union fields.

  // start with NIL
  result.operands().resize(1);
  const exprt &initializer=value.op0();
  assert(initializer.operands().size()==1);
  const irep_idt &component_name = initializer.component_name();

  // Work out what field we're initializing to. This is required, because we
  // can't work out just from the initialization expression what type the
  // operand is.

  if (!union_type.has_component(component_name))
  {
    err_location(initializer);
    str << "failed to find component `" << component_name << "'";
    throw 0;
  }

  unsigned number = union_type.component_number(component_name);
  assert(number < union_type.components().size());
  const typet &operand_type = union_type.components()[number].type();

  result.op0() = do_initializer_rec(initializer.op0(), operand_type,
                                    force_constant);

  return result;
}
