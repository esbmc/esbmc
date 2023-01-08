#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <util/c_sizeof.h>
#include <util/destructor.h>
#include <util/std_expr.h>
#include <clang-cpp-frontend/expr2cpp.h>
#include <clang-cpp-frontend/cpp_name.h>

clang_cpp_adjust::clang_cpp_adjust(contextt &_context)
  : clang_c_adjust(_context)
{
}

bool clang_cpp_adjust::adjust()
{
  clang_c_adjust::adjust();

  // additional adjustment for C++ class types

  Forall_symbol_list(it, symbol_list)
  {
    symbolt &symbol = **it;
    if(!symbol.is_type)
      continue;

    typet &type = symbol.type;
    if(type.id() == "struct")
    {
      if(type.get_bool("#class"))
      {
        // adjust class type to move functions to type's method vector
        adjust_class_type(type);
      }
    }
  }

  return false;
}

void clang_cpp_adjust::adjust_expr(exprt &expr)
{
  // special cases for C++ exprt adjustment
  // redirect everything else to C's exprt adjustment
  if(expr.id() == "ptrmember")
  {
    adjust_ptrmember(expr);
  }
  else if(expr.id() == "already_typechecked")
  {
    adjust_cpp_already_checked(expr);
  }
  else if(expr.id() == "cpp-this")
  {
    adjust_cpp_this(expr);
  }
  else
    clang_c_adjust::adjust_expr(expr);
}

void clang_cpp_adjust::adjust_cpp_this(exprt &expr)
{
  // `cpp-this` is just a placeholder. We need to get the actual exprt
  // Maps the conversion flow as in cpp_typecheckt::typecheck_expr_this

  // find `this` argument in the symbol table
  const symbolt *this_symb = namespacet(context).lookup(expr.get("#this_arg"));
  assert(this_symb);

  // now make an expression to replace `cpp-this`
  exprt this_expr("symbol", this_symb->type);
  this_expr.identifier(this_symb->id);
  this_expr.location() = this_symb->location;
  expr.swap(this_expr);
}

void clang_cpp_adjust::adjust_side_effect(side_effect_exprt &expr)
{
  const irep_idt &statement = expr.statement();

  if(statement == "cpp_new" || statement == "cpp_new[]")
  {
    adjust_new(expr);
  }
  else if(statement == "cpp_delete" || statement == "cpp_delete[]")
  {
    // adjust side effect node to explicitly call class destructor
    // e.g. the adjustment here will add the following instruction in GOTO:
    // FUNCTION_CALL:  ~t2(&(*p))
    code_function_callt destructor = get_destructor(ns, expr.type());
    if(destructor.is_not_nil())
    {
      exprt new_object("new_object", expr.type());
      new_object.cmt_lvalue(true);

      destructor.arguments().push_back(address_of_exprt(new_object));
      expr.set("destructor", destructor);
    }
  }
  else if(statement == "temporary_object")
  {
    exprt &initializer = static_cast<exprt &>(expr.add("initializer"));

    side_effect_expr_function_callt &constructor_call =
      to_side_effect_expr_function_call(initializer.op0());

    adjust_function_call_arguments(constructor_call);
  }
  else
    clang_c_adjust::adjust_side_effect(expr);
}

void clang_cpp_adjust::adjust_new(exprt &expr)
{
  if(expr.initializer().is_not_nil())
  {
    exprt &initializer = static_cast<exprt &>(expr.add("initializer"));
    adjust_expr(initializer);
  }

  // Set sizeof and cmt_sizeof_type
  exprt size_of = c_sizeof(expr.type().subtype(), ns);
  size_of.set("#c_sizeof_type", expr.type().subtype());

  expr.set("sizeof", size_of);
}

void clang_cpp_adjust::adjust_class_type(typet &type)
{
  // make a tmp struct type with empty component and method vectors
  struct_typet tmp(to_struct_type(type));
  tmp.components().clear();
  tmp.methods().clear();

  for(auto &comp : to_struct_type(type).components())
  {
    if(comp.id().as_string() == "component")
    {
      // found a field
      tmp.components().push_back(comp);
    }
    else if(comp.is_code())
    {
      // found a method: change id to "component"
      comp.id("component");
      tmp.methods().push_back(comp);
    }
  }

  type.swap(tmp);
}

void clang_cpp_adjust::adjust_ptrmember(exprt &expr)
{
  // Maps the conversion flow as in cpp_typecheckt::typecheck_expr_ptrmember
  // adjust pointer-to-member expression:
  //  e.g. this->vptr
  assert(expr.is_not_nil());

  if(expr.operands().size() != 1)
  {
    log_error("ptrmember operator expects one operand");
    abort();
  }

  // TODO: add implicit dereference, could be part of adjust_operands

  // TODO: continue from here - match expr for ptrmember!
  expr.id("member");
  clang_c_adjust::adjust_member(to_member_expr(expr));

  // make sure it's `symbol->member` where `symbol` refers to a class/struct/union object...
  if(
    expr.op0().type().id() == "symbol" &&
    expr.op0().op0().type().id() != "pointer")
  {
    log_error(
      "ptrmember operator requires pointer type on left hand side, but got "
      "`{}`",
      expr.op0().type().id().as_string());
    abort();
  }

  // get the type of `member` expr
  gen_member_type(expr, cpp_typecheck_fargst());
}

void clang_cpp_adjust::gen_member_type(
  exprt &expr,
  const cpp_typecheck_fargst &fargs)
{
  // Maps the conversion flow in cpp_typecheckt::typecheck_expr_member for C++ member
  // The conversion flow in clang_c_adjust::adjust_member
  // does not fit in our C++ conversion flows
  assert(expr.operands().size() == 1);

  exprt &op0 = expr.op0();
  // TODO: add_implicit_dereference(op0)?

  // The notation for explicit calls to destructors can be used regardless
  // of whether the type defines a destructor.  This allows you to make such
  // explicit calls without knowing if a destructor is defined for the type.
  // An explicit call to a destructor where none is defined has no effect.

  if(
    expr.find("component_cpp_name").is_not_nil() &&
    to_cpp_name(expr.find("component_cpp_name")).is_destructor() &&
    namespacet(context).follow(op0.type()).id() != "struct")
  {
    log_error("TODO: adjust expr to be \"cpp_dummy_destructor\"");
    abort();
  }

  if(op0.type().id() != "symbol")
  {
    log_error(
      "error: member operator requires type symbol on left hand side but got "
      "`{}`",
      type2cpp(op0.type(), ns));
    abort();
  }

  typet op_type = op0.type();
  // Follow symbolic types up until the last one.
  while(namespacet(context).lookup(op_type.identifier())->type.id() == "symbol")
    op_type = namespacet(context).lookup(op_type.identifier())->type;

  const irep_idt &struct_identifier = to_symbol_type(op_type).get_identifier();

  const symbolt &struct_symbol = *namespacet(context).lookup(struct_identifier);

  if(
    struct_symbol.type.id() == "incomplete_struct" ||
    struct_symbol.type.id() == "incomplete_union" ||
    struct_symbol.type.id() == "incomplete_class")
  {
    log_error("error: member operator got incomplete type on left hand side");
    abort();
  }

  if(struct_symbol.type.id() != "struct" && struct_symbol.type.id() != "union")
  {
    log_error(
      "error: member operator requires struct/union type on left hand side but "
      "got `{}`",
      type2cpp(struct_symbol.type, ns));
    abort();
  }

  //const struct_typet &type = to_struct_type(struct_symbol.type);

  if(expr.find("component_cpp_name").is_not_nil())
  {
    cpp_namet component_cpp_name = to_cpp_name(expr.find("component_cpp_name"));

    // TODO: need cpp_save_scopet?

    // resolve the member name in this scope
    cpp_typecheck_fargst new_fargs(fargs);
    new_fargs.add_object(op0);

    log_error(
      "TODO: add conversion flow to deal with `component_cpp_name` type in {}",
      __func__);
    abort();
  }

  const irep_idt &component_name = expr.component_name();

  assert(component_name != "");

  exprt component;
  component.make_nil();

  assert(
    namespacet(context).follow(expr.op0().type()).id() == "struct" ||
    namespacet(context).follow(expr.op0().type()).id() == "union");

  exprt member;

  if(get_component(expr.location(), expr.op0(), component_name, member))
  {
    // because of possible anonymous members
    expr.swap(member);
  }
  else
  {
    log_error(
      "error: member `{}` of `{}` not found",
      component_name.as_string(),
      struct_symbol.name.as_string());
    abort();
  }

  // TODO: add_implicit_dereference(op0)?

  if(expr.type().id() == "code")
  {
    log_error(
      "TODO: add conversion flow to deal with `code` type in {}", __func__);
    abort();
  }
}

bool clang_cpp_adjust::get_component(
  const locationt &location,
  const exprt &object,
  const irep_idt &component_name,
  exprt &member)
{
  // maps to the conversion flow in cpp_typecheckt::get_component
  struct_typet final_type =
    to_struct_type(namespacet(context).follow(object.type()));

  const struct_typet::componentst &components = final_type.components();

  for(const auto &component : components)
  {
    exprt tmp("member", component.type());
    tmp.component_name(component.get_name());
    tmp.location() = location;
    tmp.copy_to_operands(object);

    if(component.get_name() == component_name)
    {
      member.swap(tmp);

#if 0
      // TODO: need to check component access?
      bool not_ok = check_component_access(component, final_type);
      if(not_ok)
      {
        if(disable_access_control)
        {
          member.set("#not_accessible", true);
          member.set("#access", component.get("access"));
        }
      }
#endif

      if(object.cmt_lvalue())
        member.set("#lvalue", true);

      if(object.type().cmt_constant() && !component.get_bool("is_mutable"))
        member.type().set("#constant", true);

      member.location() = location;

      return true; // component found
    }

    if(namespacet(context)
         .follow(component.type())
         .find("#unnamed_object")
         .is_not_nil())
    {
      log_error(
        "TODO: add conversion flow to deal with `#unnamed_object` type in {}",
        __func__);
      abort();
    }
  }

  return false; // component not found
}

void clang_cpp_adjust::adjust_cpp_already_checked(exprt &expr)
{
  assert(expr.id() == "already_typechecked");
  assert(expr.operands().size() == 1);
  exprt tmp;
  tmp.swap(expr.op0());
  expr.swap(tmp);
}

void clang_cpp_adjust::adjust_side_effect_function_call(
  side_effect_expr_function_callt &expr)
{
  clang_c_adjust::adjust_side_effect_function_call(expr);
#if 0
  // For virtual functions, it is important to check whether
  // the function name is qualified. If it is qualified, then
  // the call is not virtual.
  bool is_qualified;

  if(expr.function().id() == "member" || expr.function().id() == "ptrmember")
  {
    const cpp_namet &cpp_name =
      to_cpp_name(expr.function().find("component_cpp_name"));
    is_qualified = cpp_name.is_qualified();
  }
  else if(expr.function().id() == "cpp-name")
  {
    log_error(
      "TODO: add conversion flow to deal with `cpp-name` type in {}", __func__);
    abort();
  }
  else
  {
    is_qualified = false;
  }

  // Backup of the original operand
  exprt op0 = expr.function();

  // Check typeid and return, we'll only check its parameters
  if(op0.has_operands())
  {
    if(op0.op0().statement() == "typeid")
    {
      log_error(
        "TODO: add conversion flow to deal with `typeid` type in {}", __func__);
      abort();
    }
  }

  // now do the function -- this has been postponed
  cpp_typecheck_fargst fargs(expr);
  // If the expression is decorated with what the 'this' object is, add it to
  // the fargst record. If it isn't available, name resolution will still work,
  // it just won't take the 'this' argument into account when overloading. (NB:
  // this is bad).
  if(expr.find("#this_expr").is_not_nil())
    fargs.add_object(static_cast<const exprt &>(expr.find("#this_expr")));

  adjust_function_expr(expr.function(), fargs);

  if(expr.function().id() == "type")
  {
    log_error(
      "TODO: add conversion flow to deal with `type` type in {}", __func__);
    abort();
  }

  if(expr.function().id() == "cast_expression")
  {
    log_error(
      "TODO: add conversion flow to deal with `cast_expression` type in {}", __func__);
    abort();
  }
  if(expr.function().id() == "cpp_dummy_destructor")
  {
    log_error(
      "TODO: add conversion flow to deal with `cpp_dummy_destructor` type in {}", __func__);
    abort();
  }

  // look at type of function

  assert(!"done side effect function call");
#endif
}

void clang_cpp_adjust::adjust_function_expr(
  exprt &expr,
  const cpp_typecheck_fargst &fargs)
{
  if(expr.id() == "cpp-name")
  {
    log_error("TODO: found expr id `cpp-name` in {}", __func__);
    abort();
  }
  else if(expr.id() == "member")
  {
    adjust_operands(expr);
    // TODO: missing fargs
    gen_member_type(expr, fargs);
  }
  else if(expr.id() == "ptrmember")
  {
    log_error("TODO: found expr id `ptrmember` in {}", __func__);
    abort();
  }
  else
    adjust_expr(expr);
}
