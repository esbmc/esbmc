#ifndef TYPECAST_H_
#define TYPECAST_H_

#include <util/namespace.h>
#include <util/std_expr.h>

extern void gen_derived_to_base_typecast(
  const namespacet &ns,
  exprt &dest,
  const typet &type,
  bool is_virtual /*, bool is_unckecked*/);

extern void gen_typecast(const namespacet &ns, exprt &dest, const typet &type);

extern void gen_typecast_bool(const namespacet &ns, exprt &dest);

extern void
gen_typecast_arithmetic(const namespacet &ns, exprt &expr1, exprt &expr2);

extern void gen_typecast_arithmetic(const namespacet &ns, exprt &expr);

#endif /* TYPECAST_H_ */
