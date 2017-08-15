/*******************************************************************\

Module: Base Type Computation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BASE_TYPE_H
#define CPROVER_BASE_TYPE_H

#include <util/irep2.h>
#include <util/migrate.h>
#include <util/namespace.h>
#include <util/type.h>
#include <util/union_find.h>

void base_type(type2tc &type, const namespacet &ns);
void base_type(expr2tc &expr, const namespacet &ns);

void base_type(typet &type, const namespacet &ns);
void base_type(exprt &expr, const namespacet &ns);

bool base_type_eq(
  const type2tc &type1,
  const type2tc &type2,
  const namespacet &ns);

bool base_type_eq(
  const expr2tc &expr1,
  const expr2tc &expr2,
  const namespacet &ns);

bool base_type_eq(
  const typet &type1,
  const typet &type2,
  const namespacet &ns);

bool base_type_eq(
  const exprt &expr1,
  const exprt &expr2,
  const namespacet &ns);

bool
is_subclass_of(const type2tc &subclass, const type2tc &superclass,
               const namespacet &ns);

class base_type_eqt
{
public:
  base_type_eqt(const namespacet &_ns):ns(_ns)
  {
  }

  bool base_type_eq(const type2tc &type1, const type2tc &type2)
  {
    identifiers.clear();
    return base_type_eq_rec(type1, type2);
  }

   bool base_type_eq(const typet &type1, const typet &type2)
  {
    identifiers.clear();
    return base_type_eq_rec(type1, type2);
  }

  bool base_type_eq(const expr2tc &expr1, const expr2tc &expr2)
  {
    identifiers.clear();
    return base_type_eq_rec(expr1, expr2);
  }

  bool base_type_eq(const exprt &expr1, const exprt &expr2)
  {
    identifiers.clear();
    return base_type_eq_rec(expr1, expr2);
  }

  virtual ~base_type_eqt() = default;

protected:
  const namespacet &ns;

  virtual bool base_type_eq_rec(const type2tc &type1, const type2tc &type2);
  virtual bool base_type_eq_rec(const expr2tc &expr1, const expr2tc &expr2);

  virtual bool base_type_eq_rec(const typet &type1, const typet &type2);
  virtual bool base_type_eq_rec(const exprt &expr1, const exprt &expr2);

  // for loop avoidance
  typedef union_find<irep_idt> identifierst;
  identifierst identifiers;
};

#endif
