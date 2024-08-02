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

/* GCC supports an extension called cast-to-union.
 * <https://gcc.gnu.org/onlinedocs/gcc/Cast-to-Union.html>
 *
 * It is an rvalue and looks like this:
 *
 *   union U { int a; float b; };
 *   int x;
 *   float y;
 *   (union U)x; // initializes component .a
 *   (union U)y; // initializes component .b
 *
 * Clang chose to also support this, but in a more lenient way extending this
 * support to bitfield components, which GCC does not - unless the bitfield's
 * width is the same as the underlying type's.
 *
 * We'll encode both here.
 */
static const struct_union_typet::componentt &
union_init_component(const struct_union_typet::componentst &u, const typet &t)
{
  assert(!u.empty());

  size_t max = u.size();
  for (size_t i = 0; i < u.size(); i++)
  {
    const struct_union_typet::componentt &c = u[i];
    const typet &s = c.type();
    if (s == t)
      return c;
    if (s.get_bool("#bitfield") && s.subtype() == t)
      if (
        max == u.size() ||
        atoi(s.width().c_str()) > atoi(u[max].type().width().c_str()))
        max = i;
  }
  if (max == u.size())
  {
    /* We should never reach here since clang frontend already checks for this
     * however... we should prevent any funny things to happen */
    log_error("Couldn't map type {} into the union", t.pretty_name());
    abort();
  }

  return u[max];
}

void clang_c_convertert::gen_typecast_to_union(
  const namespacet &ns,
  exprt &e,
  const typet &t)
{
  // If RHS is already of same union type, don't do anything
  if (e.type() == t.type())
    return;

  union_exprt new_result(t);
  auto &component =
    union_init_component(to_union_type(ns.follow(t)).components(), e.type());
  // Set the operator and component
  new_result.set_component_name(component.name());
  gen_typecast(ns, e, component.type());
  new_result.copy_to_operands(e);
  e.swap(new_result);
}
