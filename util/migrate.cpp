#include <stdint.h>

#include "migrate.h"
#include "prefix.h"

#include <config.h>
#include <simplify_expr.h>

// File for old irep -> new irep conversions.


void
real_migrate_type(const typet &type, type2tc &new_type_ref)
{

  if (type.id() == "bool") {
    bool_type2t *b = new bool_type2t();
    new_type_ref = type2tc(b);
  } else if (type.id() == "signedbv") {
    irep_idt width = type.width();
    unsigned int iwidth = strtol(width.as_string().c_str(), NULL, 10);
    signedbv_type2t *s = new signedbv_type2t(iwidth);
    new_type_ref = type2tc(s);
  } else if (type.id() == "unsignedbv") {
    irep_idt width = type.width();
    unsigned int iwidth = strtol(width.as_string().c_str(), NULL, 10);
    unsignedbv_type2t *s = new unsignedbv_type2t(iwidth);
    new_type_ref = type2tc(s);
  } else if (type.id() == "c_enum" || type.id() == "incomplete_c_enum") {
    // 6.7.2.2.3 of C99 says enumeration values shall have "int" types.
    signedbv_type2t *s = new signedbv_type2t(config.ansi_c.int_width);
    new_type_ref = type2tc(s);
  } else if (type.id() == "array") {
    type2tc subtype;
    expr2tc size((expr2t *)NULL);
    bool is_infinite = false;

    migrate_type(type.subtype(), subtype);

    if (type.find("size").id() == "infinity") {
      is_infinite = true;
    } else {
      exprt sz = (exprt&)type.find("size");
      simplify(sz);
      migrate_expr(sz, size);
    }

    array_type2t *a = new array_type2t(subtype, size, is_infinite);
    new_type_ref = type2tc(a);
  } else if (type.id() == "pointer") {
    type2tc subtype;

    migrate_type(type.subtype(), subtype);

    pointer_type2t *p = new pointer_type2t(subtype);
    new_type_ref = type2tc(p);
  } else if (type.id() == "empty") {
    empty_type2t *e = new empty_type2t();
    new_type_ref = type2tc(e);
  } else if (type.id() == "symbol") {
    symbol_type2t *s = new symbol_type2t(type.identifier());
    new_type_ref = type2tc(s);
  } else if (type.id() == "struct") {
    std::vector<type2tc> members;
    std::vector<std::string> names;
    struct_typet &strct = (struct_typet&)type;
    struct_union_typet::componentst comps = strct.components();

    for (struct_union_typet::componentst::const_iterator it = comps.begin();
         it != comps.end(); it++) {
      type2tc ref;
      migrate_type((const typet&)it->type(), ref);

      members.push_back(ref);
      names.push_back(it->get("name").as_string());
    }

    std::string name = type.get_string("tag");
    assert(name != "");
    struct_type2t *s = new struct_type2t(members, names, name);
    new_type_ref = type2tc(s);
  } else if (type.id() == "union") {
    std::vector<type2tc> members;
    std::vector<std::string> names;
    union_typet &strct = (union_typet&)type;
    struct_union_typet::componentst comps = strct.components();

    for (struct_union_typet::componentst::const_iterator it = comps.begin();
         it != comps.end(); it++) {
      type2tc ref;
      migrate_type((const typet&)it->type(), ref);

      members.push_back(ref);
      names.push_back(it->get("name").as_string());
    }

    std::string name = type.get_string("tag");
    assert(name != "");
    union_type2t *u = new union_type2t(members, names, name);
    new_type_ref = type2tc(u);
  } else if (type.id() == "fixedbv") {
    std::string fract = type.get_string("width");
    assert(fract != "");
    unsigned int frac_bits = strtol(fract.c_str(), NULL, 10);

    std::string ints = type.get_string("integer_bits");
    assert(ints != "");
    unsigned int int_bits = strtol(ints.c_str(), NULL, 10);

    fixedbv_type2t *f = new fixedbv_type2t(frac_bits, int_bits);
    new_type_ref = type2tc(f);
  } else if (type.id() == "code") {
    code_type2t *c = new code_type2t();
    new_type_ref = type2tc(c);
  } else {
    type.dump();
    assert(0);
  }
}

void
migrate_type(const typet &type, type2tc &new_type_ref)
{

  if (type.id() == "bool") {
    new_type_ref = type_pool.get_bool();
  } else if (type.id() == "signedbv") {
    new_type_ref = type_pool.get_signedbv(type);
  } else if (type.id() == "unsignedbv") {
    new_type_ref = type_pool.get_unsignedbv(type);
  } else if (type.id() == "c_enum" || type.id() == "incomplete_c_enum") {
    // 6.7.2.2.3 of C99 says enumeration values shall have "int" types.
    new_type_ref = type_pool.get_int(config.ansi_c.int_width);
  } else if (type.id() == "array") {
    new_type_ref = type_pool.get_array(type);
  } else if (type.id() == "pointer") {
    new_type_ref = type_pool.get_pointer(type);
  } else if (type.id() == "empty") {
    new_type_ref = type_pool.get_empty();
  } else if (type.id() == "symbol") {
    new_type_ref = type_pool.get_symbol(type);
  } else if (type.id() == "struct") {
    new_type_ref = type_pool.get_struct(type);
  } else if (type.id() == "union") {
    new_type_ref = type_pool.get_union(type);
  } else if (type.id() == "fixedbv") {
    new_type_ref = type_pool.get_fixedbv(type);
  } else if (type.id() == "code") {
    new_type_ref = type_pool.get_code();
  } else {
    type.dump();
    assert(0);
  }
}

static void
splice_expr(const exprt &expr, expr2tc &new_expr_ref)
{
  exprt expr_twopart = expr;
  exprt expr_recurse = expr;

  exprt popped = expr_recurse.operands()[expr_recurse.operands().size()-1];
  expr_recurse.operands().pop_back();

  expr_twopart.operands().clear();
  expr_twopart.copy_to_operands(expr_recurse, popped);

  migrate_expr(expr_twopart, new_expr_ref);
}

static void
convert_operand_pair(const exprt expr, expr2tc &arg1, expr2tc &arg2)
{

  migrate_expr(expr.op0(), arg1);
  migrate_expr(expr.op1(), arg2);
}

void
migrate_expr(const exprt &expr, expr2tc &new_expr_ref)
{
  type2tc type;

  if (expr.id() == "symbol") {
    migrate_type(expr.type(), type);
    expr2t *new_expr = new symbol2t(type, expr.identifier().as_string());
    new_expr_ref = expr2tc(new_expr);
  } else if (expr.id() == "nondet_symbol") {
    migrate_type(expr.type(), type);
    expr2t *new_expr = new symbol2t(type,
                                    "nondet$" + expr.identifier().as_string());
    new_expr_ref = expr2tc(new_expr);
  } else if (expr.id() == "constant" && expr.type().id() != "pointer" &&
             expr.type().id() != "bool" && expr.type().id() != "c_enum" &&
             expr.type().id() != "fixedbv" && expr.type().id() != "array") {
    migrate_type(expr.type(), type);

    bool is_signed = false;
    if (type->type_id == type2t::signedbv_id)
      is_signed = true;

    mp_integer val = binary2integer(expr.value().as_string(), is_signed);

    expr2t *new_expr = new constant_int2t(type, val);
    new_expr_ref = expr2tc(new_expr);
  } else if (expr.id() == "constant" && expr.type().id() == "c_enum") {
    migrate_type(expr.type(), type);

    uint64_t enumval = atoi(expr.value().as_string().c_str());

    expr2t *new_expr = new constant_int2t(type, BigInt(enumval));
    new_expr_ref = expr2tc(new_expr);
  } else if (expr.id() == "constant" && expr.type().id() == "bool") {
    std::string theval = expr.value().as_string();
    if (theval == "true")
      new_expr_ref = expr2tc(new constant_bool2t(true));
    else
      new_expr_ref = expr2tc(new constant_bool2t(false));
  } else if (expr.id() == "constant" && expr.type().id() == "pointer" &&
             expr.value() == "NULL") {
    // Null is a symbol with pointer type.
     migrate_type(expr.type(), type);

    expr2t *new_expr = new symbol2t(type, std::string("NULL"));
    new_expr_ref = expr2tc(new_expr);
  } else if (expr.id() == "constant" && expr.type().id() == "fixedbv") {
    migrate_type(expr.type(), type);

    fixedbvt bv(expr);

    expr2t *new_expr = new constant_fixedbv2t(type, bv);
    new_expr_ref = expr2tc(new_expr);
  } else if (expr.id() == "typecast") {
    assert(expr.op0().id_string() != "");
    expr2tc old_expr;

    migrate_type(expr.type(), type);

    migrate_expr(expr.op0(), old_expr);

    typecast2t *t = new typecast2t(type, old_expr);
    new_expr_ref = expr2tc(t);
  } else if (expr.id() == "struct") {
    migrate_type(expr.type(), type);

    std::vector<expr2tc> members;
    forall_operands(it, expr) {
      expr2tc new_ref;
      migrate_expr(*it, new_ref);

      members.push_back(new_ref);
    }

    constant_struct2t *s = new constant_struct2t(type, members);
    new_expr_ref = expr2tc(s);
  } else if (expr.id() == "union") {
    migrate_type(expr.type(), type);

    std::vector<expr2tc> members;
    forall_operands(it, expr) {
      expr2tc new_ref;
      migrate_expr(*it, new_ref);

      members.push_back(new_ref);
    }

    constant_union2t *u = new constant_union2t(type, members);
    new_expr_ref = expr2tc(u);
  } else if (expr.id() == "string-constant") {
    std::string thestring = expr.value().as_string();
    typet thetype = expr.type();
    assert(thetype.add("size").id() == "constant");
    exprt &face = (exprt&)thetype.add("size");
    std::string thelen = face.value().as_string();
    mp_integer val = binary2integer(thelen, false);

    type2tc t = type2tc(new string_type2t(val.to_long()));

    new_expr_ref = expr2tc(new constant_string2t(t, thestring));
  } else if (expr.id() == "constant" && expr.type().id() == "array") {
    // Fixed size array.
    migrate_type(expr.type(), type);

    std::vector<expr2tc> members;
    forall_operands(it, expr) {
      expr2tc new_ref;
      migrate_expr(*it, new_ref);

      members.push_back(new_ref);
    }

    constant_array2t *a = new constant_array2t(type, members);
    new_expr_ref = expr2tc(a);
  } else if (expr.id() == "array_of") {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 1);
    expr2tc new_value;
    migrate_expr(expr.op0(), new_value);

    constant_array_of2t *a = new constant_array_of2t(type, new_value);
    new_expr_ref = expr2tc(a);
  } else if (expr.id() == "if") {
    migrate_type(expr.type(), type);

    expr2tc cond, true_val, false_val;
    migrate_expr(expr.op0(), cond);
    migrate_expr(expr.op1(), true_val);
    migrate_expr(expr.op2(), false_val);

    if2t *i = new if2t(type, cond, true_val, false_val);
    new_expr_ref = expr2tc(i);
  } else if (expr.id() == "=") {
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    equality2t *e = new equality2t(side1, side2);
    new_expr_ref = expr2tc(e);
  } else if (expr.id() == "!=" || expr.id() == "notequal") {
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    notequal2t *n = new notequal2t(side1, side2);
    new_expr_ref = expr2tc(n);
   } else if (expr.id() == "<") {
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    lessthan2t *n = new lessthan2t(side1, side2);
    new_expr_ref = expr2tc(n);
   } else if (expr.id() == ">") {
    expr2tc side1, side2;
    migrate_expr(expr.op0(), side1);
    migrate_expr(expr.op1(), side2);

    greaterthan2t *n = new greaterthan2t(side1, side2);
    new_expr_ref = expr2tc(n);
  } else if (expr.id() == "<=") {
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    lessthanequal2t *n = new lessthanequal2t(side1, side2);
    new_expr_ref = expr2tc(n);
  } else if (expr.id() == ">=") {
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    greaterthanequal2t *n = new greaterthanequal2t(side1, side2);
    new_expr_ref = expr2tc(n);
  } else if (expr.id() == "not") {
    assert(expr.type().id() == "bool");
    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    not2t *n = new not2t(theval);
    new_expr_ref = expr2tc(n);
  } else if (expr.id() == "and") {
    assert(expr.type().id() == "bool");
    expr2tc side1, side2;
    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    and2t *a = new and2t(side1, side2);
    new_expr_ref = expr2tc(a);
  } else if (expr.id() == "or") {
    assert(expr.type().id() == "bool");
    expr2tc side1, side2;

    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    or2t *o = new or2t(side1, side2);
    new_expr_ref = expr2tc(o);
  } else if (expr.id() == "xor") {
    assert(expr.type().id() == "bool");
    assert(expr.operands().size() == 2);
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    xor2t *x = new xor2t(side1, side2);
    new_expr_ref = expr2tc(x);
  } else if (expr.id() == "=>") {
    assert(expr.type().id() == "bool");
    assert(expr.operands().size() == 2);
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    implies2t *i = new implies2t(side1, side2);
    new_expr_ref = expr2tc(i);
  } else if (expr.id() == "bitand") {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitand2t *a = new bitand2t(type, side1, side2);
    new_expr_ref = expr2tc(a);
  } else if (expr.id() == "bitor") {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitor2t *o = new bitor2t(type, side1, side2);
    new_expr_ref = expr2tc(o);
  } else if (expr.id() == "bitxor") {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitxor2t *x = new bitxor2t(type, side1, side2);
    new_expr_ref = expr2tc(x);
  } else if (expr.id() == "bitnand") {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitnand2t *n = new bitnand2t(type, side1, side2);
    new_expr_ref = expr2tc(n);
  } else if (expr.id() == "bitnor") {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitnor2t *o = new bitnor2t(type, side1, side2);
    new_expr_ref = expr2tc(o);
  } else if (expr.id() == "bitnxor") {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitnxor2t *x = new bitnxor2t(type, side1, side2);
    new_expr_ref = expr2tc(x);
  } else if (expr.id() == "lshr") {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    lshr2t *s = new lshr2t(type, side1, side2);
    new_expr_ref = expr2tc(s);
  } else if (expr.id() == "unary-") {
    migrate_type(expr.type(), type);

    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    neg2t *n = new neg2t(type, theval);
    new_expr_ref = expr2tc(n);
  } else if (expr.id() == "abs") {
    migrate_type(expr.type(), type);

    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    abs2t *a = new abs2t(type, theval);
    new_expr_ref = expr2tc(a);
  } else if (expr.id() == "+") {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    add2t *a = new add2t(type, side1, side2);
    new_expr_ref = expr2tc(a);
  } else if (expr.id() == "-") {
    migrate_type(expr.type(), type);

    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    sub2t *s = new sub2t(type, side1, side2);
    new_expr_ref = expr2tc(s);
  } else if (expr.id() == "*") {
    migrate_type(expr.type(), type);

    if (expr.operands().size() > 2) {
      splice_expr(expr, new_expr_ref);
      return;
    }

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    mul2t *s = new mul2t(type, side1, side2);
    new_expr_ref = expr2tc(s);
  } else if (expr.id() == "/") {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    div2t *d = new div2t(type, side1, side2);
    new_expr_ref = expr2tc(d);
  } else if (expr.id() == "mod") {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    modulus2t *m = new modulus2t(type, side1, side2);
    new_expr_ref = expr2tc(m);
  } else if (expr.id() == "shl") {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    shl2t *s = new shl2t(type, side1, side2);
    new_expr_ref = expr2tc(s);
  } else if (expr.id() == "ashr") {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    ashr2t *a = new ashr2t(type, side1, side2);
    new_expr_ref = expr2tc(a);
  } else if (expr.id() == "pointer_offset") {
    migrate_type(expr.type(), type);

    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    pointer_offset2t *p = new pointer_offset2t(type, theval);
    new_expr_ref = expr2tc(p);
  } else if (expr.id() == "pointer_object") {
    migrate_type(expr.type(), type);

    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    pointer_object2t *p = new pointer_object2t(type, theval);
    new_expr_ref = expr2tc(p);
  } else if (expr.id() == "address_of") {
    assert(expr.type().id() == "pointer");

    migrate_type(expr.type().subtype(), type);

    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    address_of2t *a = new address_of2t(type, theval);
    new_expr_ref = expr2tc(a);
   } else if (expr.id() == "byte_extract_little_endian" ||
             expr.id() == "byte_extract_big_endian") {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    bool big_endian = (expr.id() == "byte_extract_big_endian") ? true : false;

    byte_extract2t *b = new byte_extract2t(type, big_endian, side1, side2);
    new_expr_ref = expr2tc(b);
  } else if (expr.id() == "byte_update_little_endian" ||
             expr.id() == "byte_update_big_endian") {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 3);

    expr2tc sourceval, offs;
    convert_operand_pair(expr, sourceval, offs);

    expr2tc update;
    migrate_expr(expr.op2(), update);

    bool big_endian = (expr.id() == "byte_update_big_endian") ? true : false;

    byte_update2t *u = new byte_update2t(type, big_endian,
                                         sourceval, offs, update);
    new_expr_ref = expr2tc(u);
  } else if (expr.id() == "with") {
    migrate_type(expr.type(), type);

    expr2tc sourcedata, idx;
    migrate_expr(expr.op0(), sourcedata);

    if (expr.op1().id() == "member_name") {
      idx = expr2tc(new constant_string2t(type2tc(new string_type2t(1)),
                                    expr.op1().get_string("component_name")));
    } else {
      migrate_expr(expr.op1(), idx);
    }

    expr2tc update;
    migrate_expr(expr.op2(), update);

    with2t *w = new with2t(type, sourcedata, idx, update);
    new_expr_ref = expr2tc(w);
  } else if (expr.id() == "member") {
    migrate_type(expr.type(), type);

    expr2tc sourcedata;
    migrate_expr(expr.op0(), sourcedata);

    member2t *m = new member2t(type, sourcedata, expr.component_name());
    new_expr_ref = expr2tc(m);
  } else if (expr.id() == "index") {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);
    expr2tc source, index;
    convert_operand_pair(expr, source, index);

    index2t *i = new index2t(type, source, index);
    new_expr_ref = expr2tc(i);
  } else if (expr.id() == "memory-leak") {
    // Memory leaks are in fact selects/indexes.
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);
    assert(expr.type().id() == "bool");
    expr2tc source, index;
    convert_operand_pair(expr, source, index);

    index2t *i = new index2t(type, source, index);
    new_expr_ref = expr2tc(i);
  } else if (expr.id() == "zero_string") {
    assert(expr.operands().size() == 1);

    expr2tc string;
    migrate_expr(expr.op0(), string);

    zero_string2t *s = new zero_string2t(string);
    new_expr_ref = expr2tc(s);
  } else if (expr.id() == "zero_string_length") {
    assert(expr.operands().size() == 1);

    expr2tc string;
    migrate_expr(expr.op0(), string);

    zero_length_string2t *s = new zero_length_string2t(string);
    new_expr_ref = expr2tc(s);
  } else if (expr.id() == "isnan") {
    assert(expr.operands().size() == 1);

    expr2tc val;
    migrate_expr(expr.op0(), val);

    isnan2t *i = new isnan2t(val);
    new_expr_ref = expr2tc(i);
  } else if (expr.id() == "width") {
    assert(expr.operands().size() == 1);
    migrate_type(expr.type(), type);

    uint64_t thewidth = type->get_width();
    type2tc inttype(new unsignedbv_type2t(config.ansi_c.int_width));
    new_expr_ref = expr2tc(new constant_int2t(inttype, BigInt(thewidth)));
  } else if (expr.id() == "same-object") {
    assert(expr.operands().size() == 2);
    assert(expr.type().id() == "bool");
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);

    same_object2t *s = new same_object2t(op0, op1);
    new_expr_ref = expr2tc(s);
  } else if (expr.id() == "invalid-object") {
    assert(expr.type().id() == "pointer");
    type2tc pointertype(new pointer_type2t(type2tc(new empty_type2t())));
    new_expr_ref = expr2tc(new symbol2t(pointertype, "INVALID"));
  } else if (expr.id() == "unary+") {
    migrate_expr(expr.op0(), new_expr_ref);
  } else if (expr.id() == "overflow-+") {
    assert(expr.type().id() == "bool");
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    expr2tc add = expr2tc(new add2t(op0->type, op0, op1)); // XXX type?
    new_expr_ref = expr2tc(new overflow2t(add));
  } else if (expr.id() == "overflow--") {
    assert(expr.type().id() == "bool");
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    expr2tc sub = expr2tc(new sub2t(op0->type, op0, op1)); // XXX type?
    new_expr_ref = expr2tc(new overflow2t(sub));
  } else if (expr.id() == "overflow-*") {
    assert(expr.type().id() == "bool");
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    expr2tc mul = expr2tc(new mul2t(op0->type, op0, op1)); // XXX type?
    new_expr_ref = expr2tc(new overflow2t(mul));
  } else if (has_prefix(expr.id_string(), "overflow-typecast-")) {
    unsigned bits = atoi(expr.id_string().c_str() + 18);
    expr2tc operand;
    migrate_expr(expr.op0(), operand);
    new_expr_ref = expr2tc(new overflow_cast2t(operand, bits));
  } else if (expr.id() == "overflow-unary-") {
    assert(expr.type().id() == "bool");
    expr2tc operand;
    migrate_expr(expr.op0(), operand);
    new_expr_ref = expr2tc(new overflow_neg2t(operand));
  } else {
    expr.dump();
    throw new std::string("migrate expr failed");
  }
}

typet
migrate_type_back(const type2tc &ref)
{

  switch (ref->type_id) {
  case type2t::bool_id:
    return bool_typet();
  case type2t::empty_id:
    return empty_typet();
  case type2t::symbol_id:
    {
    const symbol_type2t &ref2 = to_symbol_type(ref);
    return symbol_typet(ref2.symbol_name);
    }
  case type2t::struct_id:
    {
    unsigned int idx;
    struct_typet thetype;
    struct_union_typet::componentst comps;
    const struct_type2t &ref2 = to_struct_type(ref);

    idx = 0;
    forall_types(it, ref2.members) {
      struct_union_typet::componentt component;
      component.type() = migrate_type_back(*it);
      component.set_name(irep_idt(ref2.member_names[idx]));
      comps.push_back(component);
      idx++;
    }

    thetype.components() = comps;
    thetype.set("tag", irep_idt(ref2.name));
    return thetype;
    }
  case type2t::union_id:
    {
    unsigned int idx;
    union_typet thetype;
    struct_union_typet::componentst comps;
    const union_type2t &ref2 = to_union_type(ref);

    idx = 0;
    forall_types(it, ref2.members) {
      struct_union_typet::componentt component;
      component.type() = migrate_type_back(*it);
      component.set_name(irep_idt(ref2.member_names[idx]));
      comps.push_back(component);
      idx++;
    }

    thetype.components() = comps;
    thetype.set("tag", irep_idt(ref2.name));
    return thetype;
    }
  case type2t::code_id:
    return code_typet();
  case type2t::array_id:
    {
    const array_type2t &ref2 = to_array_type(ref);

    array_typet thetype;
    thetype.subtype() = migrate_type_back(ref2.subtype);
    if (ref2.size_is_infinite) {
      thetype.set("size", "infinity");
    } else {
      thetype.size() = migrate_expr_back(ref2.array_size);
    }

    return thetype;
    }
  case type2t::pointer_id:
    {
    const pointer_type2t &ref2 = to_pointer_type(ref);

    typet subtype = migrate_type_back(ref2.subtype);
    pointer_typet thetype(subtype);
    return thetype;
    }
  case type2t::unsignedbv_id:
    {
    const unsignedbv_type2t &ref2 = to_unsignedbv_type(ref);

    return unsignedbv_typet(ref2.width);
    }
  case type2t::signedbv_id:
    {
    const signedbv_type2t &ref2 = to_signedbv_type(ref);

    return signedbv_typet(ref2.width);
    }
  case type2t::fixedbv_id:
    {
    const fixedbv_type2t &ref2 = to_fixedbv_type(ref);

    fixedbv_typet thetype;
    thetype.set_integer_bits(ref2.integer_bits);
    thetype.set("width", ref2.width);
    return thetype;
    }
  case type2t::string_id:
    return string_typet();
  default:
    assert(0 && "Unrecognized type in migrate_type_back");
  }
}

exprt
migrate_expr_back(const expr2tc &ref)
{

  if (ref.get() == NULL)
    return nil_exprt();

  switch (ref->expr_id) {
  case expr2t::constant_int_id:
  {
    const constant_int2t &ref2 = to_constant_int2t(ref);
    typet thetype = migrate_type_back(ref->type);
    constant_exprt theexpr(thetype);
    unsigned int width = atoi(thetype.width().as_string().c_str());
    theexpr.set_value(integer2binary(ref2.constant_value, width));
    return theexpr;
  }
  case expr2t::constant_fixedbv_id:
  {
    const constant_fixedbv2t &ref2 = to_constant_fixedbv2t(ref);
    return ref2.value.to_expr();
  }
  case expr2t::constant_bool_id:
  {
    const constant_bool2t &ref2 = to_constant_bool2t(ref);
    if (ref2.constant_value)
      return true_exprt();
    else
      return false_exprt();
  }
  case expr2t::constant_string_id:
  {
    const constant_string2t &ref2 = to_constant_string2t(ref);
    exprt thestring("string-constant", string_typet());
    thestring.set("value", irep_idt(ref2.value));
    const string_type2t &typeref = to_string_type(ref->type);
    constant_exprt sizeexpr(unsignedbv_typet(32));
    sizeexpr.set("value", typeref.elements);
    thestring.size(sizeexpr);
    return thestring;
  }
  case expr2t::constant_struct_id:
  {
    const constant_struct2t &ref2 = to_constant_struct2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt thestruct("struct", thetype);
    forall_exprs(it, ref2.datatype_members) {
      exprt tmp = migrate_expr_back(*it);
      thestruct.operands().push_back(tmp);
    }
    return thestruct;
  }
  case expr2t::constant_union_id:
  {
    const constant_union2t &ref2 = to_constant_union2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt theunion("union", thetype);
    forall_exprs(it, ref2.datatype_members) {
      exprt tmp = migrate_expr_back(*it);
      theunion.operands().push_back(tmp);
    }
    return theunion;
  }
  case expr2t::constant_array_id:
  {
    const constant_array2t &ref2 = to_constant_array2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt thearray("array", thetype);
    forall_exprs(it, ref2.datatype_members) {
      exprt tmp = migrate_expr_back(*it);
      thearray.operands().push_back(tmp);
    }
    return thearray;
  }
  case expr2t::constant_array_of_id:
  {
    const constant_array_of2t &ref2 = to_constant_array_of2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt thearray("array_of", thetype);
    exprt initializer = migrate_expr_back(ref2.initializer);
    thearray.operands().push_back(initializer);
    return thearray;
  }
  case expr2t::symbol_id:
  {
    const symbol2t &ref2 = to_symbol2t(ref);
    typet thetype = migrate_type_back(ref->type);
    return symbol_exprt(ref2.name, thetype);
  }
  case expr2t::typecast_id:
  {
    const typecast2t &ref2 = to_typecast2t(ref);
    typet thetype = migrate_type_back(ref->type);
    return typecast_exprt(migrate_expr_back(ref2.from), thetype);
  }
  case expr2t::if_id:
  {
    const if2t &ref2 = to_if2t(ref);
    typet thetype = migrate_type_back(ref->type);
    if_exprt theif(migrate_expr_back(ref2.cond),
                   migrate_expr_back(ref2.true_value),
                   migrate_expr_back(ref2.false_value));
    theif.type() = thetype;
    return theif;
  }
  case expr2t::equality_id:
  {
    const equality2t &ref2 = to_equality2t(ref);
    return equality_exprt(migrate_expr_back(ref2.side_1),
                          migrate_expr_back(ref2.side_2));
  }
  case expr2t::notequal_id:
  {
    const notequal2t &ref2 = to_notequal2t(ref);
    exprt notequal("notequal", bool_typet());
    notequal.copy_to_operands(migrate_expr_back(ref2.side_1),
                              migrate_expr_back(ref2.side_2));
    return notequal;
  }
  case expr2t::lessthan_id:
  {
    const lessthan2t &ref2 = to_lessthan2t(ref);
    exprt lessthan("<", bool_typet());
    lessthan.copy_to_operands(migrate_expr_back(ref2.side_1), 
                              migrate_expr_back(ref2.side_2));
    return lessthan;
  }
  case expr2t::greaterthan_id:
  {
    const greaterthan2t &ref2 = to_greaterthan2t(ref);
    exprt greaterthan(">", bool_typet());
    greaterthan.copy_to_operands(migrate_expr_back(ref2.side_1),
                              migrate_expr_back(ref2.side_2));
    return greaterthan;
  }
  case expr2t::lessthanequal_id:
  {
    const lessthanequal2t &ref2 = to_lessthanequal2t(ref);
    exprt lessthanequal("<=", bool_typet());
    lessthanequal.copy_to_operands(migrate_expr_back(ref2.side_1),
                                   migrate_expr_back(ref2.side_2));
    return lessthanequal;
  }
  case expr2t::greaterthanequal_id:
  {
    const greaterthanequal2t &ref2 = to_greaterthanequal2t(ref);
    exprt greaterthanequal(">=", bool_typet());
    greaterthanequal.copy_to_operands(migrate_expr_back(ref2.side_1),
                                      migrate_expr_back(ref2.side_2));
    return greaterthanequal;
  }
  case expr2t::not_id:
  {
    const not2t &ref2 = to_not2t(ref);
    return not_exprt(migrate_expr_back(ref2.value));
  }
  case expr2t::and_id:
  {
    const and2t &ref2 = to_and2t(ref);
    exprt andval("and", bool_typet());
    andval.copy_to_operands(migrate_expr_back(ref2.side_1),
                            migrate_expr_back(ref2.side_2));
    return andval;
  }
  case expr2t::or_id:
  {
    const or2t &ref2 = to_or2t(ref);
    exprt orval("or", bool_typet());
    orval.copy_to_operands(migrate_expr_back(ref2.side_1),
                           migrate_expr_back(ref2.side_2));
    return orval;
  }
  case expr2t::xor_id:
  {
    const xor2t &ref2 = to_xor2t(ref);
    exprt xorval("xor", bool_typet());
    xorval.copy_to_operands(migrate_expr_back(ref2.side_1),
                            migrate_expr_back(ref2.side_2));
    return xorval;
  }
  case expr2t::bitand_id:
  {
    const bitand2t &ref2 = to_bitand2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitandval("bitand", thetype);
    bitandval.copy_to_operands(migrate_expr_back(ref2.side_1),
                               migrate_expr_back(ref2.side_2));
    return bitandval;
  }
  case expr2t::bitor_id:
  {
    const bitor2t &ref2 = to_bitor2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitorval("bitor", thetype);
    bitorval.copy_to_operands(migrate_expr_back(ref2.side_1),
                               migrate_expr_back(ref2.side_2));
    return bitorval;
  }
  case expr2t::bitxor_id:
  {
    const bitxor2t &ref2 = to_bitxor2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitxorval("bitxor", thetype);
    bitxorval.copy_to_operands(migrate_expr_back(ref2.side_1),
                               migrate_expr_back(ref2.side_2));
    return bitxorval;
  }
  case expr2t::bitnand_id:
  {
    const bitnand2t &ref2 = to_bitnand2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitnandval("bitnand", thetype);
    bitnandval.copy_to_operands(migrate_expr_back(ref2.side_1),
                                migrate_expr_back(ref2.side_2));
    return bitnandval;
  }
  case expr2t::bitnor_id:
  {
    const bitnor2t &ref2 = to_bitnor2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitnorval("bitnor", thetype);
    bitnorval.copy_to_operands(migrate_expr_back(ref2.side_1),
                               migrate_expr_back(ref2.side_2));
    return bitnorval;
  }
  case expr2t::bitnxor_id:
  {
    const bitnxor2t &ref2 = to_bitnxor2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitnxorval("bitnxor", thetype);
    bitnxorval.copy_to_operands(migrate_expr_back(ref2.side_1),
                                migrate_expr_back(ref2.side_2));
    return bitnxorval;
  }
  case expr2t::lshr_id:
  {
    const lshr2t &ref2 = to_lshr2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt lshrval("lshr", thetype);
    lshrval.copy_to_operands(migrate_expr_back(ref2.side_1),
                             migrate_expr_back(ref2.side_2));
    return lshrval;
  }
  case expr2t::neg_id:
  {
    const neg2t &ref2 = to_neg2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt neg("unary-", thetype);
    neg.copy_to_operands(migrate_expr_back(ref2.value));
    return neg;
  }
  case expr2t::abs_id:
  {
    const abs2t &ref2 = to_abs2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt abs("abs", thetype);
    abs.copy_to_operands(migrate_expr_back(ref2.value));
    return abs;
  }
  case expr2t::add_id:
  {
    const add2t &ref2 = to_add2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt addval("+", thetype);
    addval.copy_to_operands(migrate_expr_back(ref2.side_1),
                            migrate_expr_back(ref2.side_1));
    return addval;
  }
  case expr2t::sub_id:
  {
    const sub2t &ref2 = to_sub2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt subval("-", thetype);
    subval.copy_to_operands(migrate_expr_back(ref2.side_1),
                            migrate_expr_back(ref2.side_1));
    return subval;
  }
  case expr2t::mul_id:
  {
    const mul2t &ref2 = to_mul2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt mulval("*", thetype);
    mulval.copy_to_operands(migrate_expr_back(ref2.side_1),
                            migrate_expr_back(ref2.side_1));
    return mulval;
  }
  case expr2t::div_id:
  {
    const div2t &ref2 = to_div2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt divval("+", thetype);
    divval.copy_to_operands(migrate_expr_back(ref2.side_1),
                            migrate_expr_back(ref2.side_1));
    return divval;
  }
  case expr2t::modulus_id:
  {
    const modulus2t &ref2 = to_modulus2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt modval("mod", thetype);
    modval.copy_to_operands(migrate_expr_back(ref2.side_1),
                            migrate_expr_back(ref2.side_1));
    return modval;
  }
  case expr2t::shl_id:
  {
    const shl2t &ref2 = to_shl2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt shlval("shl", thetype);
    shlval.copy_to_operands(migrate_expr_back(ref2.side_1),
                            migrate_expr_back(ref2.side_1));
    return shlval;
  }
  case expr2t::ashr_id:
  {
    const ashr2t &ref2 = to_ashr2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt ashrval("ashr", thetype);
    ashrval.copy_to_operands(migrate_expr_back(ref2.side_1),
                            migrate_expr_back(ref2.side_1));
    return ashrval;
  }
  case expr2t::dynamic_object_id:
    assert(0 && "dynamic_object EUNIMPLEMENTED");
  case expr2t::same_object_id:
  {
    const same_object2t &ref2 = to_same_object2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt same_objectval("same-object", thetype);
    same_objectval.copy_to_operands(migrate_expr_back(ref2.side_1),
                                    migrate_expr_back(ref2.side_1));
    return same_objectval;
  }
  case expr2t::pointer_offset_id:
  {
    const pointer_offset2t &ref2 = to_pointer_offset2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt pointer_offsetval("pointer_offset", thetype);
    pointer_offsetval.copy_to_operands(migrate_expr_back(ref2.ptr_obj));
    return pointer_offsetval;
  }
  case expr2t::pointer_object_id:
  {
    const pointer_object2t &ref2 = to_pointer_object2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt pointer_objectval("pointer_object", thetype);
    pointer_objectval.copy_to_operands(migrate_expr_back(ref2.ptr_obj));
    return pointer_objectval;
  }
  case expr2t::address_of_id:
  {
    const address_of2t &ref2 = to_address_of2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt address_ofval("address_of", thetype);
    address_ofval.copy_to_operands(migrate_expr_back(ref2.ptr_obj));
    return address_ofval;
  }
  case expr2t::byte_extract_id:
  {
    const byte_extract2t &ref2 = to_byte_extract2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt byte_extract((ref2.big_endian) ? "byte_extract_big_endian"
                                         : "byte_extract_little_endian",
                       thetype);
    byte_extract.copy_to_operands(migrate_expr_back(ref2.source_value),
                                  migrate_expr_back(ref2.source_offset));
    return byte_extract;
  }
  case expr2t::byte_update_id:
  {
    const byte_update2t &ref2 = to_byte_update2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt byte_update((ref2.big_endian) ? "byte_update_big_endian"
                                        : "byte_update_little_endian",
                       thetype);
    byte_update.copy_to_operands(migrate_expr_back(ref2.source_value),
                                  migrate_expr_back(ref2.source_offset),
                                  migrate_expr_back(ref2.update_value));
    return byte_update;
  }
  case expr2t::with_id:
  {
    const with2t &ref2 = to_with2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt with("with", thetype);
    with.copy_to_operands(migrate_expr_back(ref2.source_value),
                                  migrate_expr_back(ref2.update_field),
                                  migrate_expr_back(ref2.update_value));
    return with;
  }
  case expr2t::member_id:
  {
    const member2t &ref2 = to_member2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt member("member", thetype);
    exprt member_name("member_name");
    member_name.set("component_name", ref2.member);
    member.copy_to_operands(migrate_expr_back(ref2.source_value), member_name);
    return member;
  }
  case expr2t::index_id:
  {
    const index2t &ref2 = to_index2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt index("index", thetype);
    index.copy_to_operands(migrate_expr_back(ref2.source_value),
                                  migrate_expr_back(ref2.index));
    return index;
  }
  case expr2t::zero_string_id:
  {
    const zero_string2t &ref2 = to_zero_string2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt zerostring("zero_string", thetype);
    zerostring.copy_to_operands(migrate_expr_back(ref2.string));
    return zerostring;
  }
  case expr2t::zero_length_string_id:
  {
    const zero_length_string2t &ref2 = to_zero_length_string2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt zerostring("zero_string_length", thetype);
    zerostring.copy_to_operands(migrate_expr_back(ref2.string));
    return zerostring;
  }
  case expr2t::isnan_id:
  {
    const isnan2t &ref2 = to_isnan2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt isnan("isnan", thetype);
    isnan.copy_to_operands(migrate_expr_back(ref2.value));
    return isnan;
  }
  case expr2t::overflow_id:
  {
    const overflow2t &ref2 = to_overflow2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt theexpr;
    theexpr.type() = thetype;
    if (is_add2t(ref2.operand)) {
      theexpr.id() == "overflow-+";
      const add2t &addref = to_add2t(ref2.operand);
      theexpr.copy_to_operands(migrate_expr_back(addref.side_1),
                               migrate_expr_back(addref.side_2));
    } else if (is_sub2t(ref2.operand)) {
      theexpr.id() == "overflow--";
      const sub2t &subref = to_sub2t(ref2.operand);
      theexpr.copy_to_operands(migrate_expr_back(subref.side_1),
                               migrate_expr_back(subref.side_2));
    } else if (is_mul2t(ref2.operand)) {
      theexpr.id() == "overflow-*";
      const mul2t &mulref = to_mul2t(ref2.operand);
      theexpr.copy_to_operands(migrate_expr_back(mulref.side_1),
                               migrate_expr_back(mulref.side_2));
    } else {
      assert(0 && "Invalid operand to overflow2t when backmigrating");
    }
    return theexpr;
  }
  case expr2t::overflow_cast_id:
  {
    const overflow_cast2t &ref2 = to_overflow_cast2t(ref);
    char buffer[32];
    snprintf(buffer, 31, "%d", ref2.bits);
    buffer[31] = '\0';

    irep_idt tmp("overflow-typecast-" + std::string(buffer));
    exprt theexpr(tmp);
    typet thetype = migrate_type_back(ref->type);
    theexpr.type() = thetype;
    theexpr.copy_to_operands(migrate_expr_back(ref2.operand));
    return theexpr;
  }
  case expr2t::overflow_neg_id:
  {
    const overflow_neg2t &ref2 = to_overflow_neg2t(ref);
    exprt theexpr("overflow-unary-");
    typet thetype = migrate_type_back(ref->type);
    theexpr.type() = thetype;
    theexpr.copy_to_operands(migrate_expr_back(ref2.operand));
    return theexpr;
  }
  default:
    assert(0 && "Unrecognized expr in migrate_expr_back");
  }
}
