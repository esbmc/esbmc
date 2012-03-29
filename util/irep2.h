#ifndef _UTIL_IREP2_H_
#define _UTIL_IREP2_H_

#include <stdarg.h>

#include <vector>

#include <boost/shared_ptr.hpp>

#include <irep.h>
#include <big-int/bigint.hh>
#include <dstring.h>

// XXXjmorse - abstract, access modifies, need consideration

class prop_convt;

class type2t;
class expr2t;
typedef boost::shared_ptr<type2t> type2tc;
typedef boost::shared_ptr<expr2t> expr2tc;

/** Base class for all types */
class type2t
{
public:
  /** Enumeration identifying each sort of type.
   *  The idea being that we might (for whatever reason) at runtime need to fall
   *  back onto identifying a type through just one field, for some reason. It's
   *  also highly useful for debugging */
  enum type_ids {
    bool_id,
    empty_id,
    symbol_id,
    struct_id,
    union_id,
    code_id,
    array_id,
    pointer_id,
    unsignedbv_id,
    signedbv_id,
    fixedbv_id,
    string_id
  };

protected:
  type2t(type_ids id);
  type2t(const type2t &ref);

public:
  virtual void convert_smt_type(prop_convt &obj, void *&arg) const = 0;
  virtual unsigned int get_width(void) const = 0;

  /** Instance of type_ids recording this types type. */
  type_ids type_id;
};

template <class derived>
class type_body : public type2t
{
protected:
  type_body(type_ids id) : type2t(id) {};
  type_body(const type_body &ref);

public:
  virtual void convert_smt_type(prop_convt &obj, void *&arg) const;
};

/** Boolean type. No additional data */
class bool_type2t : public type_body<bool_type2t>
{
public:
  bool_type2t(void);
  virtual unsigned int get_width(void) const;
protected:
  bool_type2t(const bool_type2t &ref);
};

/** Empty type. For void pointers and the like, with no type. No extra data */
class empty_type2t : public type_body<empty_type2t>
{
public:
  empty_type2t(void);
  virtual unsigned int get_width(void) const;
protected:
  empty_type2t(const empty_type2t &ref);
};

/** Symbol type. Temporary, prior to linking up types after parsing, or when
 *  a struct/array contains a recursive pointer to its own type. */
class symbol_type2t : public type_body<symbol_type2t>
{
public:
  symbol_type2t(const dstring sym_name);
  virtual unsigned int get_width(void) const;
protected:
  symbol_type2t(const symbol_type2t &ref);

public:
  const dstring symbol_name;
};

class struct_union_type2t : public type_body<struct_union_type2t>
{
protected:
  struct_union_type2t(type_ids id, const std::vector<type2tc> &members);
  struct_union_type2t(const struct_union_type2t &ref);

public:
  const std::vector<type2tc> members;
};

class bv_type2t : public type_body<bv_type2t>
{
protected:
  bv_type2t(type2t::type_ids id, unsigned int width);
  bv_type2t(const bv_type2t &ref);

public:
  virtual unsigned int get_width(void) const;
  const unsigned int width;
};

// A short interlude for classes that inherit not directly from type2t - we
// need to inherit from type_body again. Except that that requires fiddling
// with the template. So, behold the below, which specializes templates that
// inherit in peculiar ways.

template <class derived>
class struct_union_type_body2t : public struct_union_type2t
{
protected:
  struct_union_type_body2t(type_ids id, const std::vector<type2tc> &members)
    : struct_union_type2t(id, members) {};
  struct_union_type_body2t(const struct_union_type_body2t &ref)
    : struct_union_type2t(ref) {};

public:
  virtual void convert_smt_type(prop_convt &obj, void *&arg) const;
};

template <class derived>
class bv_type_body : public bv_type2t
{
protected:
  bv_type_body(type_ids id, unsigned int width) : bv_type2t(id, width) {};
  bv_type_body(const bv_type_body &ref) : bv_type2t(ref) {};

public:
  virtual void convert_smt_type(prop_convt &obj, void *&arg) const;
};

class struct_type2t : public struct_union_type_body2t<struct_type2t>
{
public:
  struct_type2t(std::vector<type2tc> &members);
  virtual unsigned int get_width(void) const;
protected:
  struct_type2t(const struct_type2t &ref);
};

class union_type2t : public struct_union_type_body2t<union_type2t>
{
public:
  union_type2t(std::vector<type2tc> &members);
  virtual unsigned int get_width(void) const;
protected:
  union_type2t(const union_type2t &ref);
};

/** Code type. No additional data whatsoever. */
class code_type2t : public type_body<code_type2t>
{
public:
  code_type2t(void);
  virtual unsigned int get_width(void) const;
protected:
  code_type2t(const code_type2t &ref);
};

/** Array type. Comes with a subtype of the array and a size that might be
 *  constant, might be nondeterministic. */
class array_type2t : public type_body<array_type2t>
{
public:
  array_type2t(const type2tc subtype, const expr2tc size, bool inf);
  virtual unsigned int get_width(void) const;
protected:
  array_type2t(const array_type2t &ref);

public:

  // Exception for invalid manipulations of an infinitely sized array. No actual
  // data stored.
  class inf_sized_array_excp {
  };

  // Exception for invalid manipultions of dynamically sized arrays. No actual
  // data stored.
  class dyn_sized_array_excp {
  };

  const type2tc subtype;
  const expr2tc array_size;
  bool size_is_infinite;
};

/** Pointer type. Simply has a subtype, of what it points to. No other
 *  attributes */
class pointer_type2t : public type_body<pointer_type2t>
{
public:
  pointer_type2t(const type2tc subtype);
  virtual unsigned int get_width(void) const;
protected:
  pointer_type2t(const pointer_type2t &ref);

public:
  const type2tc subtype;
};

class unsignedbv_type2t : public bv_type_body<unsignedbv_type2t>
{
public:
  unsignedbv_type2t(unsigned int width);
protected:
  unsignedbv_type2t(const unsignedbv_type2t &ref);
};

class signedbv_type2t : public bv_type_body<signedbv_type2t>
{
public:
  signedbv_type2t(unsigned int width);
protected:
  signedbv_type2t(const signedbv_type2t &ref);
};

class fixedbv_type2t : public type_body<fixedbv_type2t>
{
public:
  fixedbv_type2t(unsigned int fraction, unsigned int integer);
  virtual unsigned int get_width(void) const;
protected:
  fixedbv_type2t(const fixedbv_type2t &ref);

public:
  const unsigned int fraction_bits;
  const unsigned int integer_bits;
};

class string_type2t : public type_body<string_type2t>
{
public:
  string_type2t(void);
  virtual unsigned int get_width(void) const;
protected:
  string_type2t(const string_type2t &ref);
};

/** Base class for all expressions */
class expr2t
{
public:
  /** Enumeration identifying each sort of expr.
   *  The idea being to permit runtime identification of a type for debugging or
   *  otherwise. See type2t::type_ids. */
  enum expr_ids {
    constant_int_id,
    constant_string_id,
    constant_struct_id,
    constant_union_id,
    constant_array_id,
    constant_array_of_id,
    symbol_id,
    typecast_id,
    if_id,
    equality_id,
    notequal_id,
    lessthan_id,
    greaterthan_id,
    lessthanequal_id,
    greaterthanequal_id,
    not_id,
    dynamic_object_id,
    is_nan_id,
    and_id,
    or_id,
    xor_id,
    same_object_id,
    bitand_id,
    bitor_id,
    bitxor_id,
    bitnand_id,
    bitnor_id,
    bitnxor_id,
    lshr_id,
    neg_id,
    abs_id,
    add_id,
    sub_id,
    mul_id,
    div_id,
    modulus_id,
    shl_id,
    ashr_id,
    pointer_offset_id,
    pointer_object_id,
    byte_extract_id,
    byte_update_id,
    with_id,
    member_id,
    index_id,
    zero_string_id,
    zero_length_string_id
  };

protected:
  expr2t(const type2tc type, expr_ids id);
  expr2t(const expr2t &ref);

public:
  /** Clone method. Entirely self explanatory */
  virtual expr2tc clone(void) const = 0;

  virtual void convert_smt(prop_convt &obj, void *&arg) const = 0;

  /** Instance of expr_ids recording tihs exprs type. */
  expr_ids expr_id;

  /** Type of this expr. All exprs have a type. */
  const type2tc type;
};

template <class derived>
class expr_body : public expr2t
{
protected:
  expr_body(const type2tc type, expr_ids id) : expr2t(type, id) {};
  expr_body(const expr_body &ref);

public:
  virtual void convert_smt(prop_convt &obj, void *&arg) const;
  virtual expr2tc clone(void) const;
};

/** Constant class type. Not designed to contain any piece of data or method in
 *  particular, but allows for additional type safety. */
class constant2t : public expr_body<constant2t>
{
public:
  constant2t(const type2tc type, expr_ids id) : expr_body<constant2t>(type, id) {}
  constant2t(const constant2t &ref) : expr_body<constant2t>(ref) {};
};

template <class T>
class const_expr_body : public constant2t
{
public:
  const_expr_body(const type2tc type, expr_ids id) : constant2t(type, id) {};
  const_expr_body(const const_expr_body &ref) : constant2t(ref) {};

  virtual void convert_smt(prop_convt &obj, void *&arg) const;
  virtual expr2tc clone(void) const;
};

/** Constant integer class. Records a constant integer of an arbitary
 *  precision */
class constant_int2t : public const_expr_body<constant_int2t>
{
public:
  constant_int2t(type2tc type, const BigInt &input);
  constant_int2t(const constant_int2t &ref);

  /** Accessor for fetching native int of this constant */
  unsigned long as_ulong(void) const;
  long as_long(void) const;

  /** Arbitary precision integer record. */
  BigInt constant_value;
};

/** Constant class for string constants. */
class constant_string2t : public constant2t
{
public:
  constant_string2t(const std::string &stringref);
protected:
  constant_string2t(constant_string2t &ref);

public:
  /** Concrete clone implementation. */
  virtual expr2tc clone(void) const;

  /** Arbitary precision integer record. */
  const std::string value;
};

/** Const datatype - for holding structs and unions */
class constant_datatype2t : public constant2t
{
public:
  constant_datatype2t(const type2tc type, const std::vector<expr2tc> &members);
protected:
  constant_datatype2t(constant_datatype2t &ref);

public:
  virtual expr2tc clone(void) const = 0;

  const std::vector<expr2tc> datatype_members;
};

class constant_struct2t : public constant_datatype2t
{
public:
  constant_struct2t(const type2tc type, const std::vector<expr2tc> &members);
protected:
  constant_struct2t(constant_struct2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class constant_union2t : public constant_datatype2t
{
public:
  constant_union2t(const type2tc type, const std::vector<expr2tc> &members);
protected:
  constant_union2t(constant_union2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class constant_array2t : public constant2t
{
public:
  constant_array2t(const type2tc type, const std::vector<expr2tc> &members);
protected:
  constant_array2t(constant_array2t &ref);

public:
  virtual expr2tc clone(void) const;

  const std::vector<expr2tc> datatype_members;
};

class constant_array_of2t : public constant2t
{
public:
  constant_array_of2t(const type2tc type, const expr2tc initializer);
protected:
  constant_array_of2t(constant_array_of2t &ref);

public:
  virtual expr2tc clone(void) const;

  // Type records the size of the array; this records the initializer.
  const expr2tc initializer;
};

class symbol2t : public expr_body<symbol2t>
{
public:
  symbol2t(const type2tc type, irep_idt name);
private:
  symbol2t(const symbol2t &ref);
  friend class expr_body<symbol2t>;

public:

  // Symbol name - only so long as a symbol is a string. In the future, this
  // should really really change.
  irep_idt name;
};

class typecast2t : public expr2t
{
public:
  typecast2t(const type2tc type, const expr2tc expr);
protected:
  typecast2t(const typecast2t &ref);

public:
  virtual expr2tc clone(void) const;

  // Expression to typecast from.
  const expr2tc from;
};

class if2t : public expr2t
{
public:
  if2t(const type2tc type, const expr2tc cond, const expr2tc true_val,
             const expr2tc false_val);
protected:
  if2t(const if2t &ref);

public:
  virtual expr2tc clone(void) const;

  // Conditional that this "if" depends on, and which value to take upon each
  // branch of that condition.
  const expr2tc cond;
  const expr2tc true_value;
  const expr2tc false_value;
};

/** Relation superclass.
 *  All subclasses should be relation operators -- ie, equality, lt, ge, so
 *  forth. Stores two expressions (of the _same_ _type_), always has result
 *  type of a bool. */
class rel2t : public expr2t
{
protected:
  rel2t(const expr2tc val1, const expr2tc val2);
  rel2t(const rel2t &ref);

public:
  virtual expr2tc clone(void) const = 0;

  const expr2tc side_1;
  const expr2tc side_2;
};

class equality2t : public rel2t
{
public:
  equality2t(const expr2tc val1, const expr2tc val2);
protected:
  equality2t(const equality2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class notequal2t : public rel2t
{
public:
  notequal2t(const expr2tc val1, const expr2tc val2);
protected:
  notequal2t(const notequal2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class lessthan2t : public rel2t
{
public:
  lessthan2t(const expr2tc val1, const expr2tc val2);
protected:
  lessthan2t(const lessthan2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class greaterthan2t : public rel2t
{
public:
  greaterthan2t(const expr2tc val1, const expr2tc val2);
protected:
  greaterthan2t(const greaterthan2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class lessthanequal2t : public rel2t
{
public:
  lessthanequal2t(const expr2tc val1, const expr2tc val2);
protected:
  lessthanequal2t(const lessthanequal2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class greaterthanequal2t : public rel2t
{
public:
  greaterthanequal2t(const expr2tc val1, const expr2tc val2);
protected:
  greaterthanequal2t(const greaterthanequal2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Logical operations base class. Base for any logical operator. No storage in
 *  this particular class. Result is always of boolean type. */
class lops2t : public expr2t
{
protected:
  lops2t(void);
  lops2t(const lops2t &ref);

public:
  virtual expr2tc clone(void) const = 0;
};

/** Not operator. Takes a boolean value; results in a boolean value. */
class not2t : public lops2t
{
public:
  not2t(const expr2tc notval);
protected:
  not2t(const not2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc notvalue;
};

/** Dynamic object operation. Checks to see whether or not the object is a
 *  dynamically allocated object or not. */
class dynamic_object2t : public lops2t
{
public:
  dynamic_object2t(const expr2tc val);
protected:
  dynamic_object2t(const dynamic_object2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc ptr_obj;
};

/** Isnan operation. Checks whether expression is a NaN or not. */
class isnan2t : public lops2t
{
public:
  isnan2t(const expr2tc val);
protected:
  isnan2t(const isnan2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc value;
};

/** Base class for 2-operand boolean oeprators. Always results in a boolean,
 *  takes two operands, both of boolean type. */
class logical_2ops2t : public lops2t
{
protected:
  logical_2ops2t(const expr2tc val1, const expr2tc val2);
  logical_2ops2t(const logical_2ops2t &ref);

public:
  virtual expr2tc clone(void) const = 0;

  const expr2tc side_1;
  const expr2tc side_2;
};

class and2t : public logical_2ops2t
{
public:
  and2t(const expr2tc val1, const expr2tc val2);
protected:
  and2t(const and2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class or2t : public logical_2ops2t
{
public:
  or2t(const expr2tc val1, const expr2tc val2);
protected:
  or2t(const or2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class xor2t : public logical_2ops2t
{
public:
  xor2t(const expr2tc val1, const expr2tc val2);
protected:
  xor2t(const xor2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Same object operation. Compares two pointer objects to see if they're the
 *  same, with a boolean result. */
class same_object2t : public logical_2ops2t
{
public:
  same_object2t(const expr2tc val1, const expr2tc val2);
protected:
  same_object2t(const same_object2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Binary operations base class. Take a type, probably integer with a width,
 *  and some operands. */
class binops2t : public expr2t
{
protected:
  binops2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  binops2t(const binops2t &ref);

public:
  virtual expr2tc clone(void) const = 0;

  const expr2tc side_1;
  const expr2tc side_2;
};

class bitand2t : public binops2t
{
public:
  bitand2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  bitand2t(const bitand2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class bitor2t : public binops2t
{
public:
  bitor2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  bitor2t(const bitor2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class bitxor2t : public binops2t
{
public:
  bitxor2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  bitxor2t(const bitxor2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class bitnand2t : public binops2t
{
public:
  bitnand2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  bitnand2t(const bitnand2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class bitnor2t : public binops2t
{
public:
  bitnor2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  bitnor2t(const bitnor2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class bitnxor2t : public binops2t
{
public:
  bitnxor2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  bitnxor2t(const bitnxor2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class lshr2t : public binops2t
{
public:
  lshr2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  lshr2t(const lshr2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Arithmatic base class. For all operations that are essentially integer
 *  arithmatic. */
class arith2t : public expr2t
{
protected:
  arith2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  arith2t(const arith2t &ref);

public:
  virtual expr2tc clone(void) const = 0;
};

class neg2t : public arith2t
{
public:
  neg2t(const type2tc type, const expr2tc value);
protected:
  neg2t(const neg2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc value;
};

class abs2t : public arith2t
{
public:
  abs2t(const type2tc type, const expr2tc value);
protected:
  abs2t(const abs2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc value;
};

/** Base two-operand arithmatic class. */
class arith_2op2t : public arith2t
{
protected:
  arith_2op2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  arith_2op2t(const arith_2op2t &ref);

public:
  virtual expr2tc clone(void) const = 0;

  const expr2tc part_1;
  const expr2tc part_2;
};

class add2t : public arith_2op2t
{
public:
  add2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  add2t(const add2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class sub2t : public arith_2op2t
{
public:
  sub2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  sub2t(const sub2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class mul2t : public arith_2op2t
{
public:
  mul2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  mul2t(const mul2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class div2t : public arith_2op2t
{
public:
  div2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  div2t(const div2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class modulus2t : public arith_2op2t
{
public:
  modulus2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  modulus2t(const modulus2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class shl2t : public arith_2op2t
{
public:
  shl2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  shl2t(const shl2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class ashr2t : public arith_2op2t
{
public:
  ashr2t(const type2tc type, const expr2tc val1, const expr2tc val2);
protected:
  ashr2t(const ashr2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Pointer offset. Extract pointer offset from a pointer value. Subclass of
 *  arithmatic because it returns an integer. */
class pointer_offset2t : public arith2t
{
public:
  pointer_offset2t(const expr2tc pointer);
protected:
  pointer_offset2t(const arith2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc pointer_obj;
};

/** Pointer object. Extract pointer object from a pointer value. Subclass of
 *  arithmatic because it returns an integer. */
class pointer_object2t : public arith2t
{
public:
  pointer_object2t(const expr2tc pointer);
protected:
  pointer_object2t(const pointer_object2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc pointer_obj;
};

/** Base class for byte operations. Endianness is a global property of the
 *  model that we're building, and we only need to care about it when we build
 *  an smt model in the end, not at any other point. */
class byte_ops2t : public expr2t
{
protected:
  byte_ops2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  byte_ops2t(const byte_ops2t &ref);

public:
  virtual expr2tc clone(void) const = 0;
};

/** Data extraction from some expression. Type is whatever type we're expecting
 *  to pull out of it. source_value is whatever piece of data we're operating
 *  upon. source_offset is the _byte_ offset into source_value to extract data
 *  from. */
class byte_extract2t : public byte_ops2t
{
public:
  byte_extract2t(const type2tc type, const expr2tc source, const expr2tc offs);
protected:
  byte_extract2t(const byte_extract2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc source_value;
  const expr2tc source_offset;
};

/** Data insertion. Type is the type of the resulting expression. source_value
 *  is the piece of data to insert data into. source_offset is the byte offset
 *  of where to put it. udpate_value is the piece of data to shoehorn into
 *  source_value. */
class byte_update2t : public byte_ops2t
{
public:
  byte_update2t(const type2tc type, const expr2tc source, const expr2tc offs,
                const expr2tc update);
protected:
  byte_update2t(const byte_update2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc source_value;
  const expr2tc source_offset;
  const expr2tc update_value;
};

/** Base type of datatype operations. */
class datatype_ops2t : public expr2t
{
protected:
  datatype_ops2t(const type2tc type);
  datatype_ops2t(const datatype_ops2t &ref);

public:
  virtual expr2tc clone(void) const = 0;
};

/** With operation. Some kind of piece of data, another piece of data to
 *  insert into it, and where to put it. */
class with2t : public datatype_ops2t
{
public:
  with2t(const type2tc type, const expr2tc source, const expr2tc update,
         const int field);
protected:
  with2t(const with2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc source_data;
  const expr2tc update_data;
  const int update_field;
};

/** Member operation. Extracts some field from a datatype. */
class member2t : public datatype_ops2t
{
public:
  member2t(const type2tc type, const expr2tc source, const int field);
protected:
  member2t(const member2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc source_data;
  const int field;
};

/** Index operation. Extracts an entry from an array. */
class index2t : public datatype_ops2t
{
public:
  index2t(const type2tc type, const expr2tc source, const expr2tc index);
protected:
  index2t(const index2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc source_data;
  const expr2tc index;
};

/** Zero string operation. Don't quite understand it. Just operates on the
 *  string struct as far as I know. Result is boolean. */
class zero_string2t : public datatype_ops2t
{
public:
  zero_string2t(const expr2tc string);
protected:
  zero_string2t(const with2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc string;
};

/** Zero length string. Unknown dirference from zero_string. */
class zero_length_string2t : public datatype_ops2t
{
public:
  zero_length_string2t(const expr2tc string);
protected:
  zero_length_string2t(const with2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc string;
};

#endif /* _UTIL_IREP2_H_ */
