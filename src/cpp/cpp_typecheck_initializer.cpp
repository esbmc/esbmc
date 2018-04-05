/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_typecheck.h>
#include <cpp/cpp_util.h>
#include <util/arith_tools.h>
#include <util/c_sizeof.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/simplify_expr_class.h>
#include <util/std_expr.h>

void cpp_typecheckt::convert_initializer(symbolt &symbol)
{
  // this is needed for template arguments that are types

  if(symbol.is_type)
  {
    if(symbol.value.is_nil())
      return;

    if(symbol.value.id() != "type")
    {
      err_location(symbol.location);
      str << "expected type as initializer for `" << symbol.base_name << "'";
      throw 0;
    }

    typecheck_type(symbol.value.type());

    return;
  }

  // do we have an initializer?
  if(symbol.value.is_nil())
  {
    // do we need one?
    if(is_reference(symbol.type))
    {
      err_location(symbol.location);
      str << "`" << symbol.base_name
          << "' is declared as reference but is not initialized";
      throw 0;
    }
    if(symbol.type.id() == "incomplete_array")
    {
      err_location(symbol.location);
      str << "storage size of `" << symbol.base_name << "' isn't know";
      throw 0;
    }

    // done
    return;
  }

  // we do have an initializer

  if(is_reference(symbol.type))
  {
    typecheck_expr(symbol.value);
    reference_initializer(symbol.value, symbol.type);
  }
  else if(cpp_is_pod(symbol.type))
  {
    if(
      symbol.type.id() == "pointer" && symbol.type.subtype().id() == "code" &&
      symbol.value.id() == "address_of" &&
      symbol.value.op0().id() == "cpp-name")
    {
      // initialization of a function pointer with
      // the address of a function: use pointer type information
      // for the sake of overload resolution

      cpp_typecheck_fargst fargs;
      fargs.in_use = true;

      const code_typet &code_type = to_code_type(symbol.type.subtype());

      for(const auto &ait : code_type.arguments())
      {
        exprt new_object("new_object");
        new_object.set("#lvalue", true);
        new_object.type() = ait.type();

        if(ait.cmt_base_name() == "this")
        {
          fargs.has_object = true;
          new_object.type() = ait.type().subtype();
        }

        fargs.operands.push_back(new_object);
      }

      exprt resolved_expr = resolve(
        to_cpp_name(static_cast<irept &>(symbol.value.op0())),
        cpp_typecheck_resolvet::BOTH,
        fargs);

      assert(symbol.type.subtype() == resolved_expr.type());

      if(resolved_expr.id() == "symbol")
      {
        symbol.value = address_of_exprt(resolved_expr);
      }
      else if(resolved_expr.id() == "member")
      {
        symbol.value =
          address_of_exprt(symbol_expr(lookup(resolved_expr.component_name())));

        symbol.value.type().add("to-member") = resolved_expr.op0().type();
      }
      else
        assert(false);

      if(symbol.type != symbol.value.type())
      {
        err_location(symbol.location);
        str << "conversion from `" << to_string(symbol.value.type()) << "' to `"
            << to_string(symbol.type) << "' ";
        throw 0;
      }

      return;
    }

    typecheck_expr(symbol.value);

    if(
      symbol.value.type().id() == "incomplete_array" ||
      symbol.value.id() == "string-constant")
    {
      do_initializer(symbol.value, symbol.type, true);
    }
    else
      implicit_typecast(symbol.value, symbol.type);

    simplify_exprt simplify;
    exprt tmp_value = symbol.value;
    if(!simplify.simplify(tmp_value))
      symbol.value.swap(tmp_value);
  }
  else
  {
    // we need a constructor

    symbol_exprt expr_symbol(symbol.name);

    exprt::operandst ops;
    ops.push_back(symbol.value);

    symbol.value = cpp_constructor(symbol.value.location(), expr_symbol, ops);
  }
}

void cpp_typecheckt::zero_initializer(
  const exprt &object,
  const typet &type,
  const locationt &location,
  exprt::operandst &ops)
{
  const typet &final_type = follow(type);

  if(final_type.id() == "struct")
  {
    std::list<codet> lst;

    forall_irep(cit, final_type.components().get_sub())
    {
      const exprt &component = static_cast<const exprt &>(*cit);

      if(component.type().id() == "code")
        continue;

      if(component.is_type())
        continue;

      if(component.get_bool("is_static"))
        continue;

      exprt member("member");
      member.copy_to_operands(object);
      member.component_name(component.name());

      // recursive call
      zero_initializer(member, component.type(), location, ops);
    }
  }
  else if(final_type.id() == "array")
  {
    if(cpp_is_pod(final_type.subtype()))
    {
      exprt value;
      c_typecheck_baset::zero_initializer(value, final_type);

      exprt obj = object;
      typecheck_expr(obj);

      code_assignt assign;
      assign.lhs() = obj;
      assign.rhs() = value;
      assign.location() = location;
      ops.push_back(assign);
    }
    else
    {
      const array_typet &array_type = to_array_type(type);
      const exprt &size_expr = array_type.size();

      if(size_expr.id() == "infinity")
        return; // don't initialize

      mp_integer size;

#ifndef NDEBUG
      bool to_int = to_integer(size_expr, size);
      assert(!to_int);
#endif
      assert(size >= 0);

      exprt::operandst empty_operands;
      for(mp_integer i = 0; i < size; ++i)
      {
        exprt index("index");
        index.copy_to_operands(object, from_integer(i, int_type()));
        zero_initializer(index, array_type.subtype(), location, ops);
      }
    }
  }
  else if(final_type.id() == "union")
  {
    // Select the largest component
    mp_integer comp_size = 0;

    exprt comp = nil_exprt();

    forall_irep(it, final_type.components().get_sub())
    {
      const exprt &component = static_cast<const exprt &>(*it);

      assert(component.type().is_not_nil());

      if(component.type().id() == "code")
        continue;

      exprt exs = c_sizeof(component.type(), *this);

      mp_integer size;
#ifndef NDEBUG
      bool to_int = !to_integer(exs, size);
      assert(to_int);
#endif

      if(size > comp_size)
      {
        comp_size = size;
        comp = component;
      }
    }

    if(comp_size > 0)
    {
      irept name("name");
      name.identifier(comp.base_name());
      name.set("#location", location);

      cpp_namet cpp_name;
      cpp_name.move_to_sub(name);

      exprt member("member");
      member.copy_to_operands(object);
      member.set("component_cpp_name", cpp_name);
      zero_initializer(member, comp.type(), location, ops);
    }
  }
  else if(final_type.id() == "c_enum")
  {
    typet enum_type("unsignedbv");
    enum_type.add("width") = final_type.find("width");

    exprt zero(gen_zero(enum_type));
    zero.make_typecast(type);
    already_typechecked(zero);

    code_assignt assign;
    assign.lhs() = object;
    assign.rhs() = zero;
    assign.location() = location;

    typecheck_expr(assign.lhs());
    assign.lhs().type().cmt_constant(false);
    already_typechecked(assign.lhs());

    typecheck_code(assign);
    ops.push_back(assign);
  }
  else if(
    final_type.id() == "incomplete_struct" ||
    final_type.id() == "incomplete_union")
  {
    err_location(location);
    str << "cannot zero-initialize incomplete compound";
    throw 0;
  }
  else
  {
    assert(gen_zero(final_type).is_not_nil());

    code_assignt assign;
    assign.lhs() = object;
    assign.rhs() = gen_zero(final_type);
    assign.location() = location;

    typecheck_expr(assign.op0());
    assign.lhs().type().cmt_constant(false);
    already_typechecked(assign.lhs());

    typecheck_code(assign);
    ops.push_back(assign);
  }
}
