#include <clang-c-frontend/typecast.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/simplify_expr_class.h>
#include <stdexcept>
#include <sstream>
#include <clang-c-frontend/clang_c_convert.h>
#include <util/message.h>

void gen_typecast(const namespacet &ns, exprt &dest, const typet &type)
{
  c_typecastt c_typecast(ns);
  c_typecast.implicit_typecast(dest, type);
}

void gen_typecast_bool(const namespacet &ns, exprt &dest)
{
  gen_typecast(ns, dest, bool_type());
}

void gen_typecast_arithmetic(const namespacet &ns, exprt &expr1, exprt &expr2)
{
  c_typecastt c_typecast(ns);
  c_typecast.implicit_typecast_arithmetic(expr1, expr2);
}

void gen_typecast_arithmetic(const namespacet &ns, exprt &expr)
{
  c_typecastt c_typecast(ns);
  c_typecast.implicit_typecast_arithmetic(expr);
}

void clang_c_convertert::gen_typecast_to_union(exprt &e, const typet &t)
{
  // If RHS is already of same union type, don't do anything
  if(e.type() == t.type())
    return;

  union_exprt new_result(t);
  for(auto component : to_union_type(t).components())
  {
    // Search for the component with the same type
    if(component.type() == e.type())
    {
      // Found it. Set the operator and component
      new_result.set_component_name(component.name());
      new_result.copy_to_operands(e);
      e.swap(new_result);
      return;
    }
  }

  /* We should never reach here since clang frontend already checks for this
   * however... we should prevent any funny things to happen */
  log_error("Couldn't map type {} into the union", e.type().pretty_name());
  abort();
}
