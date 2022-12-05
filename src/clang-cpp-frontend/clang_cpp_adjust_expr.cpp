#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <util/c_sizeof.h>
#include <util/destructor.h>

clang_cpp_adjust::clang_cpp_adjust(contextt &_context)
  : clang_c_adjust(_context)
{
}

void clang_cpp_adjust::adjust_type(typet &type)
{
  // special cases for C++ typet adjustment
  // redirect everything else to C's typet adjustment
  if(type.id() == "struct")
  {
    if(type.get_bool("#class"))
    {
      // adjust class type to move functions to type's method vector
      adjust_class_type(type);
    }
  }
  else
    clang_c_adjust::adjust_type(type);
}

void clang_cpp_adjust::adjust_expr(exprt &expr)
{
  // special cases for C++ exprt adjustment
  // redirect everything else to C's exprt adjustment
  if(expr.id() == "ptrmember")
  {
    printf("@@ cool\n");
    adjust_ptrmember(expr);
  }
  else if(expr.id() == "already_typechecked")
  {
    assert(!"Got already_typechecked for C++ exprt adjustment");
  }
  else
  {
    clang_c_adjust::adjust_expr(expr);
  }
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
  assert(expr.is_not_nil());
  assert(!"Got ptrmember");
}
