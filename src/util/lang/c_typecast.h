#ifndef CPROVER_C_TYPECAST_H
#define CPROVER_C_TYPECAST_H

#include <util/irep/expr.h>
#include <util/symtab/namespace.h>

// try a type cast from expr.type() to type
//
// false: typecast successfull, expr modified
// true:  typecast failed

bool check_c_implicit_typecast(const typet &src_type, const typet &dest_type);

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
  exprt &expr1,
  exprt &expr2,
  const namespacet &ns);

bool c_implicit_typecast_arithmetic(
  expr2tc &expr1,
  expr2tc &expr2,
  const namespacet &ns);

class c_typecastt
{
public:
  c_typecastt(const namespacet &_ns)
    : ns(_ns), no_simplify(config.options.get_bool_option("no-simplify"))
  {
  }

  virtual ~c_typecastt() = default;

  virtual void implicit_typecast(exprt &expr, const typet &type);

  virtual void implicit_typecast(expr2tc &expr, const type2tc &type);

  virtual void implicit_typecast_arithmetic(exprt &expr);

  virtual void implicit_typecast_arithmetic(expr2tc &expr);

  virtual void implicit_typecast_arithmetic(exprt &expr1, exprt &expr2);

  virtual void implicit_typecast_arithmetic(expr2tc &expr1, expr2tc &expr2);

  std::list<std::string> errors;
  std::list<std::string> warnings;

protected:
  const namespacet &ns;
  bool no_simplify;

  enum c_typet
  {
    BOOL,
    CHAR,
    UCHAR,
    INT,
    UINT,
    LONG,
    ULONG,
    LONGLONG,
    ULONGLONG,
    // Integer rank, however wide, sits below every floating type
    // (C17 6.3.1.8) and below PTR: pointer arithmetic converts neither
    // operand (6.5.6).
    INT128,
    UINT128,
    // TR 18037 fixed-point: ranks above the integers (an integer operand
    // converts toward fixed) and below the floats (a fixed operand converts
    // toward float). The rank alone doesn't identify the format, so the
    // two-operand conversion handles FIXED specially.
    FIXED,
    SINGLE,
    DOUBLE,
    LONGDOUBLE,
    VOIDPTR,
    PTR,
    OTHER
  };

  c_typet get_c_type(const typet &type);
  c_typet get_c_type(const type2tc &type);

  // Shared by both get_c_type overloads so the two copies cannot drift on the
  // width buckets, which is the divergence unit/util/c_typecast.test.cpp pins.
  static c_typet rank_integer(unsigned width, bool is_signed);
  static c_typet rank_floating(unsigned width);

  void implicit_typecast_arithmetic(exprt &expr, c_typet c_type);

  void implicit_typecast_arithmetic(expr2tc &expr, c_typet c_type);

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
