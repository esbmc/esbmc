/*******************************************************************\

Module: Conversion of sizeof Expressions

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/c_sizeof.h>
#include <ansi-c/c_typecast.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/irep2.h>
#include <util/irep2_utils.h>
#include <util/migrate.h>
#include <util/simplify_expr.h>
#include <util/type_byte_size.h>

exprt c_sizeof(const typet &src, const namespacet &ns)
{
  type2tc t;
  typet src1 = ns.follow(src);
  migrate_type(src1, t, &ns, false);

  // Array size simplification and so forth will have already occurred in
  // migration, but we might still run into a nondeterministically sized
  // array.
  mp_integer size;
  try {
    size = type_byte_size(t);
  } catch (array_type2t::dyn_sized_array_excp *e) { // Nondet'ly sized.
    std::cerr << "Sizeof nondeterministically sized array encountered"
              << std::endl;
    abort();
  } catch (array_type2t::inf_sized_array_excp *e) {
    std::cerr << "Sizeof infinite sized array encountered" << std::endl;
    abort();
  } catch (type2t::symbolic_type_excp *e) {
    std::cerr << "Sizeof symbolic type encountered" << std::endl;
    abort();
  }

  constant_int2tc theval(get_uint32_type(), size);
  return migrate_expr_back(theval);
}
