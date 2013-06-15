/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_C_TYPECAST_H
#define CPROVER_C_TYPECAST_H

#include <util/expr.h>
#include <util/namespace.h>

// try a type cast from expr.type() to type
//
// false: typecast successfull, expr modified
// true:  typecast failed

bool check_c_implicit_typecast(
  const typet &src_type,
  const typet &dest_type);

bool check_c_implicit_typecast(
  const type2tc &src_type,
  const type2tc &dest_type);

bool check_c_implicit_typecast(
  const typet &src_type,
  const typet &dest_type,
  const namespacet &ns);

bool check_c_implicit_typecast(
  const type2tc &src_type,
  const type2tc &dest_type,
  const namespacet &ns);

bool c_implicit_typecast(
  exprt &expr,
  const typet &dest_type,
  const namespacet &ns);

bool c_implicit_typecast(
  expr2tc &expr,
  const type2tc &dest_type,
  const namespacet &ns);

bool c_implicit_typecast_arithmetic(
  exprt &expr1, exprt &expr2,
  const namespacet &ns);

bool c_implicit_typecast_arithmetic(
  expr2tc &expr1, expr2tc &expr2,
  const namespacet &ns);

class c_typecastt
{
public:
  c_typecastt(const namespacet &_ns):ns(_ns)
  {
  }

  virtual ~c_typecastt() = default;

  virtual void implicit_typecast(
    exprt &expr,
    const typet &type);

  virtual void implicit_typecast(
    expr2tc &expr,
    const type2tc &type);

  virtual void implicit_typecast_arithmetic(
    exprt &expr);

  virtual void implicit_typecast_arithmetic(
    expr2tc &expr);

  virtual void implicit_typecast_arithmetic(
    exprt &expr1,
    exprt &expr2);

  virtual void implicit_typecast_arithmetic(
    expr2tc &expr1,
    expr2tc &expr2);

  std::list<std::string> errors;
  std::list<std::string> warnings;

protected:
  const namespacet &ns;

  enum c_typet { BOOL, CHAR, UCHAR, INT, UINT, LONG, ULONG,
                 LONGLONG, ULONGLONG,
                 SINGLE, DOUBLE, LONGDOUBLE,
                 VOIDPTR, PTR, OTHER };

  c_typet get_c_type(const typet &type);
  c_typet get_c_type(const type2tc &type);

  void implicit_typecast_arithmetic(
    exprt &expr,
    c_typet c_type);

  void implicit_typecast_arithmetic(
    expr2tc &expr,
    c_typet c_type);

  typet follow_with_qualifiers(const typet &src);

  type2tc follow_with_qualifiers(const type2tc &src);

  // after follow_with_qualifiers
  virtual void implicit_typecast_followed(
    exprt &expr,
    const typet &src_type,
    const typet &dest_type);

  virtual void implicit_typecast_followed(
    expr2tc &expr,
    const type2tc &src_type,
    const type2tc &dest_type);

  void do_typecast(exprt &dest, const typet &type);

  void do_typecast(expr2tc &dest, const type2tc &type);
};

#endif
