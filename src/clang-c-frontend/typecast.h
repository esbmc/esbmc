#ifndef TYPECAST_H_
#define TYPECAST_H_

#include <util/namespace.h>
#include <util/std_expr.h>

extern void gen_typecast(const namespacet &ns, exprt &dest, const typet &type);

extern void gen_typecast_bool(const namespacet &ns, exprt &dest);

extern void
gen_typecast_arithmetic(const namespacet &ns, exprt &expr1, exprt &expr2);

extern void gen_typecast_arithmetic(const namespacet &ns, exprt &expr);

/**
 * @brief Perform the typecast by creating a tmp variable on RHS
 *
 * The idea is to look for all components of the union and match
 * the type. If not found, throws an error
 *
 * @param dest RHS dest
 * @param type Union type
 */
extern void gen_typecast_to_union(exprt &dest, const typet &type);

#endif /* TYPECAST_H_ */
