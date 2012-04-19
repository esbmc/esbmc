#include "migrate.h"

#include <config.h>
#include <simplify_expr.h>

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
  } else if (type.id() == "c_enum" || type.id() == "incomplete_c_enum") {
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
      exprt sz = (exprt&)type.find("size");
      simplify(sz);
      if (!migrate_expr(sz, size))
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
  } else if (type.id() == "struct") {
    std::vector<type2tc> members;
    std::vector<std::string> names;
    struct_typet &strct = (struct_typet&)type;
    struct_union_typet::componentst comps = strct.components();

    for (struct_union_typet::componentst::const_iterator it = comps.begin();
         it != comps.end(); it++) {
      type2tc ref;
      if (!migrate_type((const typet&)it->type(), ref))
        return false;

      members.push_back(ref);
      names.push_back(it->get("name").as_string());
    }

    std::string name = type.get_string("tag");
    assert(name != "");
    struct_type2t *s = new struct_type2t(members, names, name);
    new_type_ref = type2tc(s);
    return true;
  } else if (type.id() == "union") {
    std::vector<type2tc> members;
    std::vector<std::string> names;
    union_typet &strct = (union_typet&)type;
    struct_union_typet::componentst comps = strct.components();

    for (struct_union_typet::componentst::const_iterator it = comps.begin();
         it != comps.end(); it++) {
      type2tc ref;
      if (!migrate_type((const typet&)it->type(), ref))
        return false;

      members.push_back(ref);
      names.push_back(it->get("name").as_string());
    }

    std::string name = type.get_string("tag");
    assert(name != "");
    union_type2t *u = new union_type2t(members, names, name);
    new_type_ref = type2tc(u);
    return true;
  } else if (type.id() == "fixedbv") {
    std::string fract = type.get_string("width");
    assert(fract != "");
    unsigned int frac_bits = strtol(fract.c_str(), NULL, 10);

    std::string ints = type.get_string("integer_bits");
    assert(ints != "");
    unsigned int int_bits = strtol(ints.c_str(), NULL, 10);

    fixedbv_type2t *f = new fixedbv_type2t(frac_bits, int_bits);
    new_type_ref = type2tc(f);
    return true;
  } else if (type.id() == "code") {
    code_type2t *c = new code_type2t();
    new_type_ref = type2tc(c);
    return true;
  }

  return false;
}

static bool
splice_expr(const exprt &expr, expr2tc &new_expr_ref)
{
  exprt expr_twopart = expr;
  exprt expr_recurse = expr;

  exprt popped = expr_recurse.operands()[expr_recurse.operands().size()-1];
  expr_recurse.operands().pop_back();

  expr_twopart.operands().clear();
  expr_twopart.copy_to_operands(expr_recurse, popped);

  if (!migrate_expr(expr_twopart, new_expr_ref))
    return false;
  return true;
}

static bool
convert_operand_pair(const exprt expr, expr2tc &arg1, expr2tc &arg2)
{

  if (!migrate_expr(expr.op0(), arg1))
    return false;
  if (!migrate_expr(expr.op1(), arg2))
    return false;

  return true;
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
  } else if (expr.id() == "nondet_symbol") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2t *new_expr = new symbol2t(type,
                                    "nondet$" + expr.identifier().as_string());
    new_expr_ref = expr2tc(new_expr);
    return true;
  } else if (expr.id() == "constant" && expr.type().id() != "pointer" &&
             expr.type().id() != "bool" && expr.type().id() != "c_enum" &&
             expr.type().id() != "fixedbv" && expr.type().id() != "array") {
    if (!migrate_type(expr.type(), type))
      return false;

    bool is_signed = false;
    if (type->type_id == type2t::signedbv_id)
      is_signed = true;

    mp_integer val = binary2integer(expr.value().as_string(), is_signed);

    expr2t *new_expr = new constant_int2t(type, val);
    new_expr_ref = expr2tc(new_expr);
    return true;
#if 0
  } else if (expr.id() == "typecast") {
    assert(expr.op0().id_string() != "");
    expr2tc old_expr;

    if (!migrate_type(expr.type(), type))
      return false;

    if (!migrate_expr(expr.op0(), old_expr))
      return false;

    typecast2t *t = new typecast2t(type, old_expr);
    new_expr_ref = expr2tc(t);
    return true;
#endif
  } else if (expr.id() == "struct") {
    if (!migrate_type(expr.type(), type))
      return false;

    std::vector<expr2tc> members;
    forall_operands(it, expr) {
      expr2tc new_ref;
      if (!migrate_expr(*it, new_ref))
        return false;

      members.push_back(new_ref);
    }

    constant_struct2t *s = new constant_struct2t(type, members);
    new_expr_ref = expr2tc(s);
    return true;
  } else if (expr.id() == "union") {
    if (!migrate_type(expr.type(), type))
      return false;

    std::vector<expr2tc> members;
    forall_operands(it, expr) {
      expr2tc new_ref;
      if (!migrate_expr(*it, new_ref))
        return false;

      members.push_back(new_ref);
    }

    constant_union2t *u = new constant_union2t(type, members);
    new_expr_ref = expr2tc(u);
    return true;
  } else if (expr.id() == "string-constant") {
    std::string thestring = expr.value().as_string();
    type2tc t = type2tc(new string_type2t()); // XXX-global static string type?

    constant_string2t *s = new constant_string2t(t, thestring);
    new_expr_ref = expr2tc(s);
    return true;
  } else if (expr.id() == "constant" && expr.type().id() == "array") {
    // Fixed size array.
    if (!migrate_type(expr.type(), type))
      return false;

    std::vector<expr2tc> members;
    forall_operands(it, expr) {
      expr2tc new_ref;
      if (!migrate_expr(*it, new_ref))
        return false;

      members.push_back(new_ref);
    }

    constant_array2t *a = new constant_array2t(type, members);
    new_expr_ref = expr2tc(a);
    return true;
  } else if (expr.id() == "array_of") {
    if (!migrate_type(expr.type(), type))
      return false;

    assert(expr.operands().size() == 1);
    expr2tc new_value;
    if (!migrate_expr(expr.op0(), new_value))
      return false;

    constant_array_of2t *a = new constant_array_of2t(type, new_value);
    new_expr_ref = expr2tc(a);
    return true;
  } else if (expr.id() == "if") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2tc cond, true_val, false_val;
    if (!migrate_expr(expr.op0(), cond))
      return false;
    if (!migrate_expr(expr.op1(), true_val))
      return false;
    if (!migrate_expr(expr.op2(), false_val))
      return false;

    if2t *i = new if2t(type, cond, true_val, false_val);
    new_expr_ref = expr2tc(i);
    return true;
  } else if (expr.id() == "=") {
    expr2tc side1, side2;

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    equality2t *e = new equality2t(side1, side2);
    new_expr_ref = expr2tc(e);
    return true;
  } else if (expr.id() == "!=") {
    expr2tc side1, side2;

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    notequal2t *n = new notequal2t(side1, side2);
    new_expr_ref = expr2tc(n);
    return true;
   } else if (expr.id() == "<") {
    expr2tc side1, side2;

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    lessthan2t *n = new lessthan2t(side1, side2);
    new_expr_ref = expr2tc(n);
    return true;
   } else if (expr.id() == ">") {
    expr2tc side1, side2;
    if (!migrate_expr(expr.op0(), side1))
      return false;
    if (!migrate_expr(expr.op1(), side2))
      return false;

    greaterthan2t *n = new greaterthan2t(side1, side2);
    new_expr_ref = expr2tc(n);
    return true;
  } else if (expr.id() == "<=") {
    expr2tc side1, side2;

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    lessthanequal2t *n = new lessthanequal2t(side1, side2);
    new_expr_ref = expr2tc(n);
    return true;
  } else if (expr.id() == ">=") {
    expr2tc side1, side2;

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    greaterthanequal2t *n = new greaterthanequal2t(side1, side2);
    new_expr_ref = expr2tc(n);
    return true;
  } else if (expr.id() == "not") {
    assert(expr.type().id() == "bool");
    expr2tc theval;
    if (!migrate_expr(expr.op0(), theval))
      return false;

    not2t *n = new not2t(theval);
    new_expr_ref = expr2tc(n);
    return true;
  } else if (expr.id() == "and") {
    assert(expr.type().id() == "bool");
    expr2tc side1, side2;
    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    and2t *a = new and2t(side1, side2);
    new_expr_ref = expr2tc(a);
    return true;
  } else if (expr.id() == "or") {
    assert(expr.type().id() == "bool");
    expr2tc side1, side2;

    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    or2t *o = new or2t(side1, side2);
    new_expr_ref = expr2tc(o);
    return true;
  } else if (expr.id() == "xor") {
    assert(expr.type().id() == "bool");
    assert(expr.operands().size() == 2);
    expr2tc side1, side2;

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    xor2t *x = new xor2t(side1, side2);
    new_expr_ref = expr2tc(x);
    return true;
  } else if (expr.id() == "bitand") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2tc side1, side2;
    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    bitand2t *a = new bitand2t(type, side1, side2);
    new_expr_ref = expr2tc(a);
    return true;
  } else if (expr.id() == "bitor") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2tc side1, side2;
    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    bitor2t *o = new bitor2t(type, side1, side2);
    new_expr_ref = expr2tc(o);
    return true;
  } else if (expr.id() == "bitxor") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2tc side1, side2;
    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    bitxor2t *x = new bitxor2t(type, side1, side2);
    new_expr_ref = expr2tc(x);
    return true;
  } else if (expr.id() == "bitnand") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2tc side1, side2;
    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    bitnand2t *n = new bitnand2t(type, side1, side2);
    new_expr_ref = expr2tc(n);
    return true;
  } else if (expr.id() == "bitnor") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2tc side1, side2;
    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    bitnor2t *o = new bitnor2t(type, side1, side2);
    new_expr_ref = expr2tc(o);
    return true;
  } else if (expr.id() == "bitnxor") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2tc side1, side2;
    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    bitnxor2t *x = new bitnxor2t(type, side1, side2);
    new_expr_ref = expr2tc(x);
    return true;
  } else if (expr.id() == "lshr") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2tc side1, side2;
    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    lshr2t *s = new lshr2t(type, side1, side2);
    new_expr_ref = expr2tc(s);
    return true;
  } else if (expr.id() == "unary-") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2tc theval;
    if (!migrate_expr(expr.op0(), theval))
      return false;

    neg2t *n = new neg2t(type, theval);
    new_expr_ref = expr2tc(n);
    return true;
  } else if (expr.id() == "abs") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2tc theval;
    if (!migrate_expr(expr.op0(), theval))
      return false;

    abs2t *a = new abs2t(type, theval);
    new_expr_ref = expr2tc(a);
    return true;
  } else if (expr.id() == "add") {
    if (!migrate_type(expr.type(), type))
      return false;

    expr2tc side1, side2;
    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    if (!convert_operand_pair(expr, side1, side2))
        return false;

    add2t *a = new add2t(type, side1, side2);
    new_expr_ref = expr2tc(a);
    return true;
  } else if (expr.id() == "sub") {
    if (!migrate_type(expr.type(), type))
      return false;

    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    expr2tc side1, side2;
    if (!convert_operand_pair(expr, side1, side2))
        return false;

    sub2t *s = new sub2t(type, side1, side2);
    new_expr_ref = expr2tc(s);
    return true;
  } else if (expr.id() == "mul") {
    if (!migrate_type(expr.type(), type))
      return false;

    if (expr.operands().size() > 2)
      return splice_expr(expr, new_expr_ref);

    expr2tc side1, side2;
    if (!convert_operand_pair(expr, side1, side2))
        return false;

    mul2t *s = new mul2t(type, side1, side2);
    new_expr_ref = expr2tc(s);
    return true;
  } else {
    return false;
  }
}
