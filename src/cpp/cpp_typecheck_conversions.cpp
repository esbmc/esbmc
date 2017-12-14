/*******************************************************************\

Module: C++ Language Type Checking

Author:

\*******************************************************************/

#include <util/c_qualifiers.h>
#include <cpp/cpp_typecheck.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/expr_util.h>
#include <util/simplify_expr_class.h>

bool cpp_typecheckt::standard_conversion_lvalue_to_rvalue(
  const exprt &expr,
  exprt &new_expr) const
{
  assert(expr.cmt_lvalue());

  if(
    expr.type().id() == "code" || expr.type().id() == "incomplete_array" ||
    expr.type().id() == "incomplete_class" ||
    expr.type().id() == "incomplete_struct" ||
    expr.type().id() == "incomplete_union")
    return false;

  new_expr = expr;
  new_expr.remove("#lvalue");

  return true;
}

bool cpp_typecheckt::standard_conversion_array_to_pointer(
  const exprt &expr,
  exprt &new_expr) const
{
  assert(expr.type().id() == "array" || expr.type().id() == "incomplete_array");

  exprt index("index", expr.type().subtype());
  index.copy_to_operands(expr, from_integer(0, int_type()));
  index.set("#lvalue", true);

  pointer_typet pointer;
  pointer.subtype() = expr.type().subtype();

  new_expr = exprt("address_of", pointer);
  new_expr.move_to_operands(index);

  return true;
}

bool cpp_typecheckt::standard_conversion_function_to_pointer(
  const exprt &expr,
  exprt &new_expr) const
{
  const code_typet &func_type = to_code_type(expr.type());

  if(!expr.cmt_lvalue())
    return false;

  pointer_typet pointer;
  pointer.subtype() = func_type;

  new_expr = exprt("address_of");
  new_expr.copy_to_operands(expr);
  new_expr.type() = pointer;

  return true;
}

bool cpp_typecheckt::standard_conversion_qualification(
  const exprt &expr,
  const typet &type,
  exprt &new_expr) const
{
  if(expr.type().id() != "pointer" || is_reference(expr.type()))
    return false;

  if(expr.cmt_lvalue())
    return false;

  if(expr.type() != type)
    return false;

  typet sub_from = expr.type().subtype();
  typet sub_to = type.subtype();
  bool const_to = true;

  while(sub_from.id() == "pointer")
  {
    c_qualifierst qual_from;
    qual_from.read(sub_from);

    c_qualifierst qual_to;
    qual_to.read(sub_to);

    if(!qual_to.is_constant)
      const_to = false;

    if(qual_from.is_constant && !qual_to.is_constant)
      return false;

    if(qual_from != qual_to && !const_to)
      return false;

    typet tmp1 = sub_from.subtype();
    sub_from.swap(tmp1);

    typet tmp2 = sub_to.subtype();
    sub_to.swap(tmp2);
  }

  c_qualifierst qual_from(sub_from);
  c_qualifierst qual_to(sub_to);

  if(qual_from.is_subset_of(qual_to))
  {
    new_expr = expr;
    new_expr.type() = type;
    return true;
  }

  return false;
}

bool cpp_typecheckt::standard_conversion_integral_promotion(
  const exprt &expr,
  exprt &new_expr) const
{
  if(expr.cmt_lvalue())
    return false;

  c_qualifierst qual_from;
  qual_from.read(expr.type());

  typet int_type("signedbv");
  int_type.width(config.ansi_c.int_width);
  qual_from.write(int_type);

  if(expr.type().id() == "signedbv")
  {
    unsigned width = to_signedbv_type(expr.type()).get_width();
    if(width >= config.ansi_c.int_width)
      return false;
    new_expr = expr;
    new_expr.make_typecast(int_type);
    return true;
  }

  if(expr.type().id() == "unsignedbv")
  {
    unsigned width = to_unsignedbv_type(expr.type()).get_width();
    if(width >= config.ansi_c.int_width)
      return false;
    new_expr = expr;
    if(width == config.ansi_c.int_width)
      int_type.id("unsignedbv");
    new_expr.make_typecast(int_type);
    return true;
  }

  if(follow(expr.type()).id() == "c_enum")
  {
    new_expr = expr;
    new_expr.make_typecast(int_type);
    return true;
  }

  return false;
}

bool cpp_typecheckt::standard_conversion_floating_point_promotion(
  const exprt &expr,
  exprt &new_expr) const
{
  if(expr.cmt_lvalue())
    return false;

  // we only do that with 'float',
  // not with 'double' or 'long double'
  if(expr.type() != float_type())
    return false;

  unsigned width = bv_width(expr.type());

  if(width != config.ansi_c.single_width)
    return false;

  c_qualifierst qual_from;
  qual_from.read(expr.type());

  new_expr = expr;
  new_expr.make_typecast(double_type());
  qual_from.write(new_expr.type());

  return true;
}

bool cpp_typecheckt::standard_conversion_integral_conversion(
  const exprt &expr,
  const typet &type,
  exprt &new_expr) const
{
  if(type.id() != "signedbv" && type.id() != "unsignedbv")
    return false;

  if(
    expr.type().id() != "signedbv" && expr.type().id() != "unsignedbv" &&
    expr.type().id() != "bool" && follow(expr.type()).id() != "c_enum")
    return false;

  if(expr.cmt_lvalue())
    return false;

  c_qualifierst qual_from;
  qual_from.read(expr.type());
  new_expr = expr;
  new_expr.make_typecast(type);
  qual_from.write(new_expr.type());

  return true;
}

bool cpp_typecheckt::standard_conversion_floating_integral_conversion(
  const exprt &expr,
  const typet &type,
  exprt &new_expr) const
{
  if(expr.cmt_lvalue())
    return false;

  if(expr.type().id() == "floatbv" || expr.type().id() == "fixedbv")
  {
    if(type.id() != "signedbv" && type.id() != "unsignedbv")
      return false;
  }
  else if(
    expr.type().id() == "signedbv" || expr.type().id() == "unsignedbv" ||
    follow(expr.type()).id() == "c_enum")
  {
    if(type.id() != "fixedbv" && type.id() != "floatbv")
      return false;
  }
  else
    return false;

  c_qualifierst qual_from;
  qual_from.read(expr.type());
  new_expr = expr;
  new_expr.make_typecast(type);
  qual_from.write(new_expr.type());

  return true;
}

bool cpp_typecheckt::standard_conversion_floating_point_conversion(
  const exprt &expr,
  const typet &type,
  exprt &new_expr) const
{
  if(expr.type().id() != "floatbv" && expr.type().id() != "fixedbv")
    return false;

  if(type.id() != "floatbv" && type.id() != "fixedbv")
    return false;

  if(expr.cmt_lvalue())
    return false;

  c_qualifierst qual_from;

  qual_from.read(expr.type());
  new_expr = expr;
  new_expr.make_typecast(type);
  qual_from.write(new_expr.type());

  return true;
}

bool cpp_typecheckt::standard_conversion_pointer(
  const exprt &expr,
  const typet &type,
  exprt &new_expr)
{
  if(
    type.id() != "pointer" || is_reference(type) ||
    type.find("to-member").is_not_nil())
    return false;

  if(expr.cmt_lvalue())
    return false;

  if(expr.is_zero())
  {
    new_expr = expr;
    new_expr.value("NULL");
    new_expr.type() = type;
    return true;
  }

  if(
    expr.type().id() != "pointer" || expr.type().find("to-member").is_not_nil())
    return false;

  typet sub_from = follow(expr.type().subtype());
  typet sub_to = follow(type.subtype());

  // anything but function pointer to void *
  if(sub_from.id() != "code" && sub_to.id() == "empty")
  {
    c_qualifierst qual_from;
    qual_from.read(expr.type().subtype());
    new_expr = expr;
    new_expr.make_typecast(type);
    qual_from.write(new_expr.type().subtype());
    return true;
  }

  // struct * to struct *
  if(sub_from.id() == "struct" && sub_to.id() == "struct")
  {
    const struct_typet &from_struct = to_struct_type(sub_from);
    const struct_typet &to_struct = to_struct_type(sub_to);
    if(subtype_typecast(from_struct, to_struct))
    {
      c_qualifierst qual_from;
      qual_from.read(expr.type().subtype());
      new_expr = expr;
      make_ptr_typecast(new_expr, type);
      qual_from.write(new_expr.type().subtype());
      return true;
    }
  }

  return false;
}

bool cpp_typecheckt::standard_conversion_pointer_to_member(
  const exprt &expr,
  const typet &type,
  exprt &new_expr)
{
  if(
    type.id() != "pointer" || is_reference(type) ||
    type.find("to-member").is_nil())
    return false;

  if(expr.type().id() != "pointer" || expr.type().find("to-member").is_nil())
    return false;

  if(type.subtype() != expr.type().subtype())
  {
    // subtypes different
    if(type.subtype().id() == "code" && expr.type().subtype().id() == "code")
    {
      code_typet code1 = to_code_type(expr.type().subtype());
      assert(code1.arguments().size() > 0);
      code_typet::argumentt this1 = code1.arguments()[0];
      assert(this1.cmt_base_name() == "this");
      code1.arguments().erase(code1.arguments().begin());

      code_typet code2 = to_code_type(type.subtype());
      assert(code2.arguments().size() > 0);
      code_typet::argumentt this2 = code2.arguments()[0];
      assert(this2.cmt_base_name() == "this");
      code2.arguments().erase(code2.arguments().begin());

      if(
        this2.type().subtype().cmt_constant() &&
        !this1.type().subtype().cmt_constant())
        return false;

      // give a second chance ignoring `this'
      if(code1 != code2)
        return false;
    }
    else
      return false;
  }

  if(expr.cmt_lvalue())
    return false;

  if(expr.id() == "constant" && expr.value() == "NULL")
  {
    new_expr = expr;
    new_expr.make_typecast(type);
    return true;
  }

  struct_typet from_struct = to_struct_type(
    follow(static_cast<const typet &>(expr.type().find("to-member"))));

  struct_typet to_struct =
    to_struct_type(follow(static_cast<const typet &>(type.find("to-member"))));

  if(subtype_typecast(to_struct, from_struct))
  {
    new_expr = expr;
    new_expr.make_typecast(type);
    return true;
  }

  return false;
}

bool cpp_typecheckt::standard_conversion_boolean(
  const exprt &expr,
  exprt &new_expr) const
{
  if(expr.cmt_lvalue())
    return false;

  if(
    expr.type().id() != "signedbv" && expr.type().id() != "unsignedbv" &&
    expr.type().id() != "pointer" && follow(expr.type()).id() != "c_enum")
    return false;

  c_qualifierst qual_from;
  qual_from.read(expr.type());

  bool_typet Bool;
  qual_from.write(Bool);

  new_expr = expr;
  new_expr.make_typecast(Bool);
  return true;
}

bool cpp_typecheckt::standard_conversion_sequence(
  const exprt &expr,
  const typet &type,
  exprt &new_expr,
  cpp_typecast_rank &rank)
{
  assert(!is_reference(expr.type()) && !is_reference(type));

  exprt curr_expr = expr;

  if(
    curr_expr.type().id() == "array" ||
    curr_expr.type().id() == "incomplete_array")
  {
    if(type.id() == "pointer")
    {
      if(!standard_conversion_array_to_pointer(curr_expr, new_expr))
        return false;
    }
  }
  else if(curr_expr.type().id() == "code" && type.id() == "pointer")
  {
    if(!standard_conversion_function_to_pointer(curr_expr, new_expr))
      return false;
  }
  else if(curr_expr.cmt_lvalue())
  {
    if(!standard_conversion_lvalue_to_rvalue(curr_expr, new_expr))
      return false;
  }
  else
    new_expr = curr_expr;

  curr_expr.swap(new_expr);

  if(curr_expr.type() != type)
  {
    if(
      type.id() == "signedbv" || type.id() == "unsignedbv" ||
      follow(type).id() == "c_enum")
    {
      if(
        !standard_conversion_integral_promotion(curr_expr, new_expr) ||
        new_expr.type() != type)
      {
        if(!standard_conversion_integral_conversion(curr_expr, type, new_expr))
        {
          if(!standard_conversion_floating_integral_conversion(
               curr_expr, type, new_expr))
            return false;
        }
        rank.rank += 3;
      }
      else
        rank.rank += 2;
    }
    else if(type.id() == "floatbv" || type.id() == "fixedbv")
    {
      if(
        !standard_conversion_floating_point_promotion(curr_expr, new_expr) ||
        new_expr.type() != type)
      {
        if(
          !standard_conversion_floating_point_conversion(
            curr_expr, type, new_expr) &&
          !standard_conversion_floating_integral_conversion(
            curr_expr, type, new_expr))
          return false;

        rank.rank += 3;
      }
      else
        rank.rank += 2;
    }
    else if(type.id() == "pointer")
    {
      if(!standard_conversion_pointer(curr_expr, type, new_expr))
      {
        if(!standard_conversion_pointer_to_member(curr_expr, type, new_expr))
          return false;
      }
      rank.rank += 3;

      // Did we just cast to a void pointer?
      if(
        type.subtype().id() == "empty" && curr_expr.type().id() == "pointer" &&
        curr_expr.type().subtype().id() != "empty")
        rank.has_ptr_to_voidptr = true;
    }
    else if(type.id() == "bool")
    {
      if(!standard_conversion_boolean(curr_expr, new_expr))
        return false;
      rank.rank += 3;

      // Pointer to bool conversion might lead to special disambiguation later
      if(curr_expr.type().id() == "pointer")
        rank.has_ptr_to_bool = true;
    }
    else
      return false;
  }
  else
    new_expr = curr_expr;

  curr_expr.swap(new_expr);

  if(curr_expr.type().id() == "pointer")
  {
    typet sub_from = curr_expr.type();
    typet sub_to = type;

    do
    {
      typet tmp_from = sub_from.subtype();
      sub_from.swap(tmp_from);
      typet tmp_to = sub_to.subtype();
      sub_to.swap(tmp_to);

      c_qualifierst qual_from;
      qual_from.read(sub_from);

      c_qualifierst qual_to;
      qual_to.read(sub_to);

      if(qual_from != qual_to)
      {
        rank.rank += 1;
        break;
      }

    } while(sub_from.id() == "pointer");

    if(!standard_conversion_qualification(curr_expr, type, new_expr))
      return false;
  }
  else
  {
    new_expr = curr_expr;
    new_expr.type() = type;
  }

  return true;
}

bool cpp_typecheckt::user_defined_conversion_sequence(
  const exprt &expr,
  const typet &type,
  exprt &new_expr,
  cpp_typecast_rank &rank)
{
  static bool recursion_guard = false;
  assert(!is_reference(expr.type()));
  assert(!is_reference(type));

  const typet &from = follow(expr.type());
  const typet &to = follow(type);

  new_expr.make_nil();

  rank.rank += 4;

  if(to.id() == "struct")
  {
    std::string err_msg;

    if(cpp_is_pod(to))
    {
      if(from.id() == "struct")
      {
        const struct_typet &from_struct = to_struct_type(from);
        const struct_typet &to_struct = to_struct_type(to);

        if(subtype_typecast(from_struct, to_struct) && expr.cmt_lvalue())
        {
          exprt address("address_of", pointer_typet());
          address.copy_to_operands(expr);
          address.type().subtype() = expr.type();

          // simplify address
          if(expr.id() == "dereference")
            address = expr.op0();

          pointer_typet ptr_sub;
          ptr_sub.subtype() = type;
          c_qualifierst qual_from;
          qual_from.read(expr.type());
          qual_from.write(ptr_sub.subtype());
          make_ptr_typecast(address, ptr_sub);

          exprt deref("dereference");
          deref.copy_to_operands(address);
          deref.type() = address.type().subtype();

          // create temporary object
          exprt tmp_object_expr = exprt("sideeffect", type);
          tmp_object_expr.statement("temporary_object");
          tmp_object_expr.location() = expr.location();
          tmp_object_expr.copy_to_operands(deref);
          tmp_object_expr.set("#lvalue", true);

          new_expr.swap(tmp_object_expr);
          return true;
        }
      }
    }
    else if(!recursion_guard)
    {
      struct_typet from_struct;
      from_struct.make_nil();

      if(from.id() == "struct")
        from_struct = to_struct_type(from);

      struct_typet to_struct = to_struct_type(to);

      // Look up a constructor that will build us a temporary of the correct
      // type, and takes an argument of the relevant type. Do this by asking
      // the resolve code to look it up for us; this avoids duplication, and
      // nets us template constructors too.

      // Move to the struct scope
      cpp_scopet &scope = cpp_scopes.get_scope(to_struct.name());
      cpp_save_scopet cpp_saved_scope(cpp_scopes);
      cpp_scopes.go_to(scope);

      // Just look up the plain name from this scope.
      // XXX this is super dodgy.
      irept thename = to_struct.add("tag").get_sub()[0];
      cpp_namet name_record;
      name_record.get_sub().push_back(thename);

      // Make a fake temporary.
      symbol_exprt fake_temp("fake_temporary", type);
      fake_temp.type().remove("#constant");
      fake_temp.cmt_lvalue(true);

      cpp_typecheck_fargst fargs;
      fargs.in_use = true;
      fargs.add_object(fake_temp);
      fargs.operands.push_back(expr);

      // Disallow more than one level of implicit construction.
      recursion_guard = true;
      exprt result =
        resolve(name_record, cpp_typecheck_resolvet::VAR, fargs, false);
      recursion_guard = false;

      if(result.is_nil())
        goto out;

      // XXX explicit?
      if(result.type().get_bool("is_explicit"))
        goto out;

      if(result.type().id() != "code")
        goto out;

      if(result.type().return_type().id() != "constructor")
        goto out;

      result.location() = expr.location();

      {
        exprt tmp("already_typechecked");
        tmp.copy_to_operands(result);
        result.swap(result);
      }

      // create temporary object
      side_effect_expr_function_callt ctor_expr;
      ctor_expr.location() = expr.location();
      ctor_expr.function().swap(result);
      ctor_expr.arguments().push_back(expr);
      typecheck_side_effect_function_call(ctor_expr);

      new_expr.swap(ctor_expr);
      assert(new_expr.statement() == "temporary_object");

      if(to.cmt_constant())
        new_expr.type().cmt_constant(true);
    }
  }
  else if(to.id() == "bool")
  {
    std::string name = expr.type().identifier().as_string();
    if(
      name == "std::tag.istream" || name == "std::tag.ostream" ||
      name == "std::tag.iostream" || name == "std::tag.ifstream")
    {
      exprt nondet_expr("nondet_symbol", bool_typet());
      new_expr.swap(nondet_expr);
    }
    else if(expr.id() == "symbol")
    {
      // Can't blindly cast aggregate / composite types. NB: Nothing here
      // actually appears to look up any custom convertors.
      if(
        expr.type().id() == "array" || expr.type().id() == "struct" ||
        expr.type().id() == "union")
        return false;

      exprt tmp_expr = expr;
      tmp_expr.make_typecast(bool_typet());
      new_expr.swap(tmp_expr);
    }
  }

out:

  // conversion operators
  if(from.id() == "struct")
  {
    struct_typet from_struct = to_struct_type(from);

    bool found = false;
    for(struct_typet::componentst::const_iterator it =
          from_struct.components().begin();
        it != from_struct.components().end();
        it++)
    {
      const irept &component = *it;
      const typet comp_type = static_cast<const typet &>(component.type());

      if(component.get_bool("from_base"))
        continue;

      if(!component.get_bool("is_cast_operator"))
        continue;

      assert(
        component.get("type") == "code" &&
        component.type().arguments().get_sub().size() == 1);

      typet this_type = static_cast<const typet &>(
        comp_type.arguments().get_sub().front().type());
      this_type.set("#reference", true);

      exprt this_expr(expr);
      this_type.set("#this", true);

      cpp_typecast_rank tmp_rank;
      exprt tmp_expr;

      if(implicit_conversion_sequence(this_expr, this_type, tmp_expr, tmp_rank))
      {
        // To take care of the possible virtual case,
        // we build the function as a member expression.
        irept func_name("name");
        func_name.identifier(component.base_name());
        cpp_namet cpp_func_name;
        cpp_func_name.get_sub().push_back(func_name);

        exprt member_func("member");
        member_func.add("component_cpp_name") = cpp_func_name;
        exprt ac("already_typechecked");
        ac.copy_to_operands(expr);
        member_func.copy_to_operands(ac);

        side_effect_expr_function_callt func_expr;
        func_expr.location() = expr.location();
        func_expr.function().swap(member_func);
        typecheck_side_effect_function_call(func_expr);

        exprt tmp_expr;
        if(standard_conversion_sequence(func_expr, type, tmp_expr, tmp_rank))
        {
          // check if it's ambiguous
          if(found)
            return false;
          found = true;

          rank += tmp_rank;
          new_expr.swap(tmp_expr);
        }
      }
    }
    if(found)
      return true;
  }

  return new_expr.is_not_nil();
}

bool cpp_typecheckt::reference_related(const exprt &expr, const typet &type)
  const
{
  assert(is_reference(type));
  assert(!is_reference(expr.type()));
  if(expr.type() == type.subtype())
    return true;

  typet from = follow(expr.type());
  typet to = follow(type.subtype());

  if(from.id() == "struct" && to.id() == "struct")
    return subtype_typecast(to_struct_type(from), to_struct_type(to));

  if(
    from.id() == "struct" && type.get_bool("#this") &&
    type.subtype().id() == "empty")
  {
    // virtual-call case
    return true;
  }

  return false;
}

bool cpp_typecheckt::reference_compatible(
  const exprt &expr,
  const typet &type,
  cpp_typecast_rank &rank) const
{
  assert(is_reference(type));
  assert(!is_reference(expr.type()));

  if(!reference_related(expr, type))
    return false;

  if(expr.type() != type.subtype())
    rank.rank += 3;

  c_qualifierst qual_from;
  qual_from.read(expr.type());

  c_qualifierst qual_to;
  qual_to.read(type.subtype());

  if(qual_from != qual_to)
    rank.rank += 1;

  if(qual_from.is_subset_of(qual_to))
    return true;

  return false;
}

bool cpp_typecheckt::reference_binding(
  exprt expr,
  const typet &type,
  exprt &new_expr,
  cpp_typecast_rank &rank)
{
  assert(is_reference(type));
  assert(!is_reference(expr.type()));

  cpp_typecast_rank backup_rank = rank;

  if(type.get_bool("#this") && !expr.cmt_lvalue())
  {
    // `this' has to be an lvalue
    if(expr.statement() == "temporary_object")
      expr.set("#lvalue", true);
    else if(expr.statement() == "function_call")
      expr.set("#lvalue", true);
    else if(expr.get_bool("#temporary_avoided"))
    {
      expr.remove("#temporary_avoided");
      exprt temporary;
      new_temporary(expr.location(), expr.type(), expr, temporary);
      expr.swap(temporary);
      expr.set("#lvalue", true);
    }
    else
      return false;
  }

  if(expr.cmt_lvalue())
  {
    // TODO: Check this
    if(reference_compatible(expr, type, rank))
    {
      if(expr.id() == "dereference")
      {
        new_expr = expr.op0();
        new_expr.type().set("#reference", true);
      }
      else
      {
        address_of_exprt tmp;
        tmp.location() = expr.location();
        tmp.object() = expr;
        tmp.type() = pointer_typet();
        tmp.type().set("#reference", true);
        tmp.type().subtype() = tmp.op0().type();
        new_expr.swap(tmp);
      }

      if(expr.type() != type.subtype())
      {
        c_qualifierst qual_from;
        qual_from.read(expr.type());
        new_expr.make_typecast(type);
        qual_from.write(new_expr.type().subtype());
      }

      return true;
    }

    rank = backup_rank;
  }

  // conversion operators
  typet from_type = follow(expr.type());
  if(from_type.id() == "struct")
  {
    struct_typet from_struct = to_struct_type(from_type);

    for(struct_typet::componentst::const_iterator it =
          from_struct.components().begin();
        it != from_struct.components().end();
        it++)
    {
      const irept &component = *it;

      if(component.get_bool("from_base"))
        continue;

      if(!component.get_bool("is_cast_operator"))
        continue;

      const code_typet &component_type =
        to_code_type(static_cast<const typet &>(component.type()));

      // otherwise it cannot bind directly (not an lvalue)
      if(!is_reference(component_type.return_type()))
        continue;

      assert(component_type.arguments().size() == 1);

      typet this_type = component_type.arguments().front().type();
      this_type.set("#reference", true);

      exprt this_expr(expr);

      this_type.set("#this", true);

      cpp_typecast_rank tmp_rank;

      exprt tmp_expr;
      if(implicit_conversion_sequence(this_expr, this_type, tmp_expr, tmp_rank))
      {
        // To take care of the possible virtual case,
        // we build the function as a member expression.
        irept func_name("name");
        func_name.identifier(component.base_name());
        cpp_namet cpp_func_name;
        cpp_func_name.get_sub().push_back(func_name);

        exprt member_func("member");
        member_func.add("component_cpp_name") = cpp_func_name;
        exprt ac("already_typechecked");
        ac.copy_to_operands(expr);
        member_func.copy_to_operands(ac);

        side_effect_expr_function_callt func_expr;
        func_expr.location() = expr.location();
        func_expr.function().swap(member_func);
        typecheck_side_effect_function_call(func_expr);

        // let's check if the returned value binds directly
        exprt returned_value = func_expr;
        add_implicit_dereference(returned_value);

        if(
          returned_value.cmt_lvalue() &&
          reference_compatible(returned_value, type, rank))
        {
          // returned values are lvalues in case of references only
          assert(
            returned_value.id() == "dereference" &&
            is_reference(returned_value.op0().type()));

          new_expr = returned_value.op0();

          if(returned_value.type() != type.subtype())
          {
            c_qualifierst qual_from;
            qual_from.read(returned_value.type());
            make_ptr_typecast(new_expr, type);
            qual_from.write(new_expr.type().subtype());
          }
          rank += tmp_rank;
          rank.rank += 4;
          return true;
        }
      }
    }
  }

  // No temporary allowed for `this'
  if(type.get_bool("#this"))
    return false;

  if(!type.subtype().cmt_constant() || type.subtype().cmt_volatile())
    return false;

  // TODO: hanlde the case for implicit parameters
  if(!type.subtype().cmt_constant() && !expr.cmt_lvalue())
    return false;

  exprt arg_expr = expr;

  if(follow(arg_expr.type()).id() == "struct")
  {
    // required to initialize the temporary
    arg_expr.set("#lvalue", true);
  }

  if(user_defined_conversion_sequence(arg_expr, type.subtype(), new_expr, rank))
  {
    address_of_exprt tmp;
    tmp.type() = pointer_typet();
    tmp.object() = new_expr;
    tmp.type().set("#reference", true);
    tmp.type().subtype() = new_expr.type();
    tmp.location() = new_expr.location();
    new_expr.swap(tmp);
    return true;
  }

  rank = backup_rank;
  if(standard_conversion_sequence(expr, type.subtype(), new_expr, rank))
  {
    {
      // create temporary object
      exprt tmp = exprt("sideeffect", type.subtype());
      tmp.statement("temporary_object");
      tmp.location() = expr.location();
      //tmp.set("#lvalue", true);
      tmp.move_to_operands(new_expr);
      new_expr.swap(tmp);
    }

    exprt tmp("address_of", pointer_typet());
    tmp.copy_to_operands(new_expr);
    tmp.type().set("#reference", true);
    tmp.type().subtype() = new_expr.type();
    tmp.location() = new_expr.location();
    new_expr.swap(tmp);
    return true;
  }

  return false;
}

bool cpp_typecheckt::implicit_conversion_sequence(
  const exprt &expr,
  const typet &type,
  exprt &new_expr,
  cpp_typecast_rank &rank)
{
  cpp_typecast_rank backup_rank = rank;

  exprt e = expr;
  add_implicit_dereference(e);

  if(is_reference(type))
  {
    if(!reference_binding(e, type, new_expr, rank))
      return false;

    simplify_exprt simplify;
    simplify.simplify(new_expr);
    new_expr.type().set("#reference", true);
  }
  else if(!standard_conversion_sequence(e, type, new_expr, rank))
  {
    rank = backup_rank;
    if(!user_defined_conversion_sequence(e, type, new_expr, rank))
      return false;

    simplify_exprt simplify;
    simplify.simplify(new_expr);
  }

  return true;
}

bool cpp_typecheckt::implicit_conversion_sequence(
  const exprt &expr,
  const typet &type,
  exprt &new_expr)
{
  cpp_typecast_rank rank;
  return implicit_conversion_sequence(expr, type, new_expr, rank);
}

bool cpp_typecheckt::implicit_conversion_sequence(
  const exprt &expr,
  const typet &type,
  cpp_typecast_rank &rank)
{
  exprt new_expr;
  return implicit_conversion_sequence(expr, type, new_expr, rank);
}

void cpp_typecheckt::implicit_typecast(exprt &expr, const typet &type)
{
  exprt e = expr;

  if(!implicit_conversion_sequence(e, type, expr))
  {
    show_instantiation_stack(str);
    err_location(e);
    str << "invalid implicit conversion from `" << to_string(e.type())
        << "' to `" << to_string(type) << "' ";
    throw 0;
  }
}

void cpp_typecheckt::reference_initializer(exprt &expr, const typet &type)
{
  assert(is_reference(type));
  add_implicit_dereference(expr);

  cpp_typecast_rank rank;
  exprt new_expr;
  if(reference_binding(expr, type, new_expr, rank))
  {
    expr.swap(new_expr);
    return;
  }

  err_location(expr);
  str << "bad reference initializer";
  throw 0;
}

bool cpp_typecheckt::cast_away_constness(const typet &t1, const typet &t2) const
{
  assert(t1.id() == "pointer" && t2.id() == "pointer");
  typet nt1 = t1;
  typet nt2 = t2;

  if(is_reference(nt1))
    nt1.remove("#reference");

  if(is_reference(nt2))
    nt2.remove("#reference");

  // substitute final subtypes
  std::vector<typet> snt1;
  snt1.push_back(nt1);

  while(snt1.back().find("subtype").is_not_nil())
  {
    snt1.reserve(snt1.size() + 1);
    snt1.push_back(snt1.back().subtype());
  }

  c_qualifierst q1;
  q1.read(snt1.back());

  bool_typet newnt1;
  q1.write(newnt1);
  snt1.back() = newnt1;

  std::vector<typet> snt2;
  snt2.push_back(nt2);
  while(snt2.back().find("subtype").is_not_nil())
  {
    snt2.reserve(snt2.size() + 1);
    snt2.push_back(snt2.back().subtype());
  }

  c_qualifierst q2;
  q2.read(snt2.back());

  bool_typet newnt2;
  q2.write(newnt2);
  snt2.back() = newnt2;

  const int k = snt1.size() < snt2.size() ? snt1.size() : snt2.size();

  for(int i = k; i > 1; i--)
  {
    snt1[snt1.size() - 2].subtype() = snt1[snt1.size() - 1];
    snt1.pop_back();

    snt2[snt2.size() - 2].subtype() = snt2[snt2.size() - 1];
    snt2.pop_back();
  }

  exprt e1("Dummy", snt1.back());
  exprt e2;

  return !standard_conversion_qualification(e1, snt2.back(), e2);
}

bool cpp_typecheckt::const_typecast(
  const exprt &expr,
  const typet &type,
  exprt &new_expr)
{
  assert(is_reference(expr.type()) == false);

  exprt curr_expr = expr;

  if(
    curr_expr.type().id() == "array" ||
    curr_expr.type().id() == "incomplete_array")
  {
    if(type.id() == "pointer")
    {
      if(!standard_conversion_array_to_pointer(curr_expr, new_expr))
        return false;
    }
  }
  else if(curr_expr.type().id() == "code" && type.id() == "pointer")
  {
    if(!standard_conversion_function_to_pointer(curr_expr, new_expr))
      return false;
  }
  else if(curr_expr.cmt_lvalue())
  {
    if(!standard_conversion_lvalue_to_rvalue(curr_expr, new_expr))
      return false;
  }
  else
    new_expr = curr_expr;

  if(is_reference(type))
  {
    if(!expr.cmt_lvalue())
      return false;

    if(new_expr.type() != type.subtype())
      return false;

    exprt address_of("address_of", type);
    address_of.copy_to_operands(expr);
    add_implicit_dereference(address_of);
    new_expr.swap(address_of);
    return true;
  }
  if(type.id() == "pointer")
  {
    if(type != new_expr.type())
      return false;

    // add proper typecast
    typecast_exprt typecast_expr(expr, type);
    new_expr.swap(typecast_expr);
    return true;
  }

  return false;
}

bool cpp_typecheckt::dynamic_typecast(
  const exprt &expr,
  const typet &type,
  exprt &new_expr)
{
  exprt e(expr);

  if(type.id() == "pointer")
  {
    if(e.id() == "dereference" && e.implicit())
      e = expr.op0();

    if(e.type().id() == "pointer" && cast_away_constness(e.type(), type))
      return false;
  }
  add_implicit_dereference(e);

  if(is_reference(type))
  {
    exprt typeid_function = new_expr;
    exprt function = typeid_function;
    irep_idt badcast_identifier = "std::tag.bad_cast";

    // We must check if the user included typeinfo
    const symbolt *bad_cast_symbol;
    bool is_included = lookup(badcast_identifier, bad_cast_symbol);

    if(is_included)
      throw "Error: must #include <typeinfo>. Bad_cast throw";

    // Ok! Let's create the temp object badcast
    exprt badcast;
    badcast.identifier(badcast_identifier);
    badcast.operands().emplace_back("sideeffect");
    badcast.op0().type() = typet("symbol");
    badcast.op0().type().identifier(badcast_identifier);

    // Check throw
    typecheck_expr_throw(badcast);

    // Save on the expression for handling on goto-program
    function.set("exception_list", badcast.find("exception_list"));
    e.make_typecast(type);
    new_expr.swap(e);
    new_expr.op0().operands().push_back(badcast);
    return true;

    if(follow(type.subtype()).id() != "struct")
    {
      return false;
    }
  }
  if(type.id() == "pointer")
  {
    if(type.find("to-member").is_not_nil())
      return false;

    if(type.subtype().id() == "empty")
    {
      if(!e.cmt_lvalue())
        return false;
    }
    else if(follow(type.subtype()).id() == "struct")
    {
      if(e.cmt_lvalue())
      {
        exprt tmp(e);

        if(!standard_conversion_lvalue_to_rvalue(tmp, e))
          return false;
      }
    }
    else
      return false;
  }
  else
    return false;

  bool res = static_typecast(e, type, new_expr);

  if(res)
  {
    if(type.id() == "pointer" && e.type().id() == "pointer")
    {
      if(type.find("to-member").is_nil() && e.type().find("to-member").is_nil())
      {
        typet to = follow(type.subtype());
        symbolt t;
        if(e.identifier() != "")
        {
          t = lookup(e.identifier());
        }
        else if(e.op0().identifier() != "") // Array
        {
          t = lookup(e.op0().identifier());
        }
        else
          return false;

        typet from;
        if(t.type.id() == "array")
        {
          if(type.id() == new_expr.type().id())
          {
            from = follow(t.value.op0().type());
            e.make_typecast(type);
            new_expr.op0().op0().operands() = t.value.operands();
            return true;
          }
        }

        from = follow(t.value.type());

        // Could not dynamic_cast from void type
        if(t.type.subtype().id() == "empty")
        {
          return false;
        }

        // Are we doing a dynamic typecast between objects of the same class type?
        if(
          (type.id() == new_expr.type().id()) &&
          (type.subtype().id() == new_expr.type().subtype().id()))
        {
          return true;
        }

        if(from.id() == "empty")
        {
          e.make_typecast(type);
          new_expr.swap(e);
          return true;
        }

        if(to.id() == "struct" && from.id() == "struct")
        {
          if(e.cmt_lvalue())
          {
            exprt tmp(e);
            if(!standard_conversion_lvalue_to_rvalue(tmp, e))
              return false;
          }

          struct_typet from_struct = to_struct_type(from);
          struct_typet to_struct = to_struct_type(to);
          if(subtype_typecast(from_struct, to_struct))
          {
            make_ptr_typecast(e, type);
            new_expr.op0().swap(t.value);
            return true;
          }
        }

        // Cannot make typecast
        constant_exprt null_expr;
        null_expr.type() = new_expr.type();
        null_expr.set_value("NULL");

        new_expr.swap(null_expr);
        return true;
      }
      if(
        type.find("to-member").is_not_nil() &&
        e.type().find("to-member").is_not_nil())
      {
        if(type.subtype() != e.type().subtype())
          return false;

        struct_typet from_struct = to_struct_type(
          follow(static_cast<const typet &>(e.type().find("to-member"))));

        struct_typet to_struct = to_struct_type(
          follow(static_cast<const typet &>(type.find("to-member"))));

        if(subtype_typecast(from_struct, to_struct))
        {
          new_expr = e;
          new_expr.make_typecast(type);
          return true;
        }
      }
      else
        return false;
    }

    return false;
  }

  err_location(expr.location());
  str << "expr: " << e << std::endl;
  str << "totype: " << type << std::endl;
  throw "Could not cast types in dynamic cast";

  return false;
}

bool cpp_typecheckt::reinterpret_typecast(
  const exprt &expr,
  const typet &type,
  exprt &new_expr,
  bool check_constantness)
{
  exprt e = expr;

  if(check_constantness && type.id() == "pointer")
  {
    if(e.id() == "dereference" && e.implicit())
      e = expr.op0();

    if(e.type().id() == "pointer" && cast_away_constness(e.type(), type))
      return false;
  }

  add_implicit_dereference(e);

  if(!is_reference(type))
  {
    exprt tmp;

    if(e.id() == "code")
    {
      if(standard_conversion_function_to_pointer(e, tmp))
        e.swap(tmp);
      else
        return false;
    }

    if(e.type().id() == "array" || e.type().id() == "incomplete_array")
    {
      if(standard_conversion_array_to_pointer(e, tmp))
        e.swap(tmp);
      else
        return false;
    }

    if(e.cmt_lvalue())
    {
      if(standard_conversion_lvalue_to_rvalue(e, tmp))
        e.swap(tmp);
      else
        return false;
    }
  }

  if(
    e.type().id() == "pointer" &&
    (type.id() == "unsignedbv" || type.id() == "signedbv"))
  {
    // pointer to integer, always ok
    new_expr = e;
    new_expr.make_typecast(type);
    return true;
  }

  if(
    (e.type().id() == "unsignedbv" || e.type().id() == "signedbv" ||
     e.type().id() == "bool") &&
    type.id() == "pointer" && !is_reference(type))
  {
    // integer to pointer
    if(e.is_zero())
    {
      // NULL
      new_expr = e;
      new_expr.value("NULL");
      new_expr.type() = type;
    }
    else
    {
      new_expr = e;
      new_expr.make_typecast(type);
    }
    return true;
  }

  if(
    e.type().id() == "pointer" && type.id() == "pointer" && !is_reference(type))
  {
    if(e.type().subtype().id() == "code" && type.subtype().id() != "code")
      return false;
    if(e.type().subtype().id() != "code" && type.subtype().id() == "code")
      return false;

    // this is more generous than the standard
    new_expr = expr;
    new_expr.make_typecast(type);
    return true;
  }

  if(is_reference(type) && e.cmt_lvalue())
  {
    exprt tmp("address_of", pointer_typet());
    tmp.type().subtype() = e.type();
    tmp.copy_to_operands(e);
    tmp.make_typecast(type);
    new_expr.swap(tmp);
    return true;
  }

  return false;
}

bool cpp_typecheckt::static_typecast(
  const exprt &expr,
  const typet &type,
  exprt &new_expr,
  bool check_constantness)
{
  exprt e = expr;

  if(check_constantness && type.id() == "pointer")
  {
    if(e.id() == "dereference" && e.implicit())
      e = expr.op0();

    if(e.type().id() == "pointer" && cast_away_constness(e.type(), type))
      return false;
  }

  add_implicit_dereference(e);

  if(type.reference())
  {
    cpp_typecast_rank rank;
    if(reference_binding(e, type, new_expr, rank))
      return true;

    typet subto = follow(type.subtype());
    typet from = follow(e.type());

    if(subto.id() == "struct" && from.id() == "struct")
    {
      if(!expr.cmt_lvalue())
        return false;

      c_qualifierst qual_from;
      qual_from.read(e.type());

      c_qualifierst qual_to;
      qual_to.read(type.subtype());

      if(!qual_to.is_subset_of(qual_from))
        return false;

      struct_typet from_struct = to_struct_type(from);
      struct_typet subto_struct = to_struct_type(subto);

      if(subtype_typecast(subto_struct, from_struct))
      {
        if(e.id() == "dereference")
        {
          make_ptr_typecast(e.op0(), type);
          new_expr.swap(e.op0());
          return true;
        }

        exprt address_of("address_of", pointer_typet());
        address_of.type().subtype() = e.type();
        address_of.copy_to_operands(e);
        make_ptr_typecast(address_of, type);
        new_expr.swap(address_of);
        return true;
      }
    }
    return false;
  }

  if(type.id() == "empty")
  {
    new_expr = e;
    new_expr.make_typecast(type);
    return true;
  }

  if(
    follow(type).id() == "c_enum" &&
    (e.type().id() == "signedbv" || e.type().id() == "unsignedbv" ||
     follow(e.type()).id() == "c_enum"))
  {
    new_expr = e;
    new_expr.make_typecast(type);
    if(new_expr.cmt_lvalue())
      new_expr.remove("#lvalue");
    return true;
  }

  if(implicit_conversion_sequence(e, type, new_expr))
  {
    if(!cpp_is_pod(type))
    {
      exprt tc("already_typechecked");
      tc.copy_to_operands(new_expr);
      exprt temporary;
      new_temporary(e.location(), type, tc, temporary);
      new_expr.swap(temporary);
    }
    else
    {
      // try to avoid temporary
      new_expr.set("#temporary_avoided", true);
      if(new_expr.cmt_lvalue())
        new_expr.remove("#lvalue");
    }

    return true;
  }

  if(type.id() == "pointer" && e.type().id() == "pointer")
  {
    if(type.find("to-member").is_nil() && e.type().find("to-member").is_nil())
    {
      typet to = follow(type.subtype());
      typet from = follow(e.type().subtype());

      if(from.id() == "empty")
      {
        e.make_typecast(type);
        new_expr.swap(e);
        return true;
      }

      if(to.id() == "struct" && from.id() == "struct")
      {
        if(e.cmt_lvalue())
        {
          exprt tmp(e);
          if(!standard_conversion_lvalue_to_rvalue(tmp, e))
            return false;
        }

        struct_typet from_struct = to_struct_type(from);
        struct_typet to_struct = to_struct_type(to);
        if(subtype_typecast(to_struct, from_struct))
        {
          make_ptr_typecast(e, type);
          new_expr.swap(e);
          return true;
        }
      }

      return false;
    }
    if(
      type.find("to-member").is_not_nil() &&
      e.type().find("to-member").is_not_nil())
    {
      if(type.subtype() != e.type().subtype())
        return false;

      struct_typet from_struct = to_struct_type(
        follow(static_cast<const typet &>(e.type().find("to-member"))));

      struct_typet to_struct = to_struct_type(
        follow(static_cast<const typet &>(type.find("to-member"))));

      if(subtype_typecast(from_struct, to_struct))
      {
        new_expr = e;
        new_expr.make_typecast(type);
        return true;
      }
    }
    else
      return false;
  }

  return false;
}
