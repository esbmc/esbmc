#include "migrate.h"

#include <config.h>

// File for old irep -> new irep conversions.


bool
migrate_type(const typet &type, type2tc &new_type_ref)
{

  if (type.id() == "bool") {
    bool_type2t *b = new bool_type2t();
    new_type_ref = type2tc(b);
    return true;
  } else if (type.id() == "signedbv") {
    irep_idt width = type.width();
    unsigned int iwidth = strtol(width.as_string().c_str(), NULL, 10);
    signedbv_type2t *s = new signedbv_type2t(iwidth);
    new_type_ref = type2tc(s);
    return true;
  } else if (type.id() == "unsignedbv") {
    irep_idt width = type.width();
    unsigned int iwidth = strtol(width.as_string().c_str(), NULL, 10);
    unsignedbv_type2t *s = new unsignedbv_type2t(iwidth);
    new_type_ref = type2tc(s);
    return true;
  } else if (type.id() == "c_enum") {
    // 6.7.2.2.3 of C99 says enumeration values shall have "int" types.
    signedbv_type2t *s = new signedbv_type2t(config.ansi_c.int_width);
    new_type_ref = type2tc(s);
    return true;
  } else if (type.id() == "array") {
    type2tc subtype;
    expr2tc size((expr2t *)NULL);
    bool is_infinite = false;

    if (!migrate_type(type.subtype(), subtype))
      return false;

    if (type.find("size").id() == "infinity") {
      is_infinite = true;
    } else {
      if (!migrate_expr((const exprt&)type.find("size"), size))
        return false;
    }

    array_type2t *a = new array_type2t(subtype, size, is_infinite);
    new_type_ref = type2tc(a);
    return true;
  } else if (type.id() == "pointer") {
    type2tc subtype;

    if (!migrate_type(type.subtype(), subtype))
      return false;

    pointer_type2t *p = new pointer_type2t(subtype);
    new_type_ref = type2tc(p);
    return true;
  } else if (type.id() == "empty") {
    empty_type2t *e = new empty_type2t();
    new_type_ref = type2tc(e);
    return true;
  } else if (type.id() == "symbol") {
    symbol_type2t *s = new symbol_type2t(type.identifier());
    new_type_ref = type2tc(s);
    return true;
  }

  return false;
}

bool
migrate_expr(const exprt &expr, expr2tc &new_expr_ref)
{
  type2tc type;

  if (expr.id() == "symbol") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2t *new_expr = new symbol2t(type, expr.identifier().as_string());
    new_expr_ref = expr2tc(new_expr);
    return true;
  } else if (expr.id() == "constant" && expr.type().id() != "pointer") {
    if (!migrate_type(expr.type(), type))
      return false;

    bool is_signed = false;
    if (type->type_id == type2t::signedbv_id)
      is_signed = true;

    mp_integer val = binary2integer(expr.value().as_string(), is_signed);

    expr2t *new_expr = new constant_int2t(type, val);
    new_expr_ref = expr2tc(new_expr);
    return true;
  } else {
    return false;
  }
}
