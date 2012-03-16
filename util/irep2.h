#include <vector>

#include <boost/shared_ptr.hpp>

#include <big-int/bigint.hh>

/** Base class for all types */
class type2t
{
protected:
  type2t(type_ids id);
  type2t(const type2t &ref);

public:
  /** Enumeration identifying each sort of type.
   *  The idea being that we might (for whatever reason) at runtime need to fall
   *  back onto identifying a type through just one field, for some reason. It's
   *  also highly useful for debugging */
  enum type_ids {
  };

  /** Instance of type_ids recording this types type. */
  type_ids type_id;
};

/** Boolean type. No additional data */
class bool_type2t : type2t
{
protected:
  bool_type2t();
  bool_type2t(const bool_type2t &ref);
};

/** Empty type. For void pointers and the like, with no type. No extra data */
class empty_type2t : type2t
{
protected:
  empty_type2t();
  empty_type2t(const empty_type2t &ref);
};

/** Symbol type. Temporary, prior to linking up types after parsing, or when
 *  a struct/array contains a recursive pointer to its own type. */
class symbol_type2t : type2t
{
protected:
  symbol_type2t(const dstring sym_name);
  symbol_type2t(const symbol_type2t &ref);

public:
  const dstring symbol_name;
};

class struct_union_type2t : typet
{
protected:
  struct_union_type2t(std::vector<const type2t *> &members);
  struct_union_type2t(const struct_union_type2t &ref);

public:
  const std::vector<const type2t *> &members;
};

class struct_type2t : struct_union_type2t
{
protected:
  struct_type2t(std::vector<const type2t *> &members);
  struct_type2t(const struct_type2t &ref);
};

class union_type2t : struct_union_type2t
{
protected:
  union_type2t(std::vector<const type2t *> &members);
  union_type2t(const union_type2t &ref);
};

/** Code type. No additional data whatsoever. */
class code_type2t : type2t
{
protected:
  code_type2t(void);
  code_type2t(const code_type2t &ref);
};

/** Array type. Comes with a subtype of the array and a size that might be
 *  constant, might be nondeterministic. */
class array_type2t : type2t
{
protected:
  array_type2t(const type2t &subtype, const expr2tc size);
  array_type2t(const array_type2t &ref);

public:
  type2t &subtype;
  expr2tc array_size;
};

/** Base class for all expressions */
class expr2t
{
protected:
  expr2t(type2t &type, expr_ids id);
  expr2t(const expr2t &ref);

public:
  /** Enumeration identifying each sort of expr.
   *  The idea being to permit runtime identification of a type, for debugging or
   *  otherwise. See type2t::type_ids. */
  enum expr_ids {
  };

  /** Clone method. Entirely self explanatory */
  virtual expr2tc clone(void) const = 0;

  /** Instance of expr_ids recording tihs exprs type. */
  expr_ids expr_id;

  /** Type of this expr. All exprs have a type. */
  const type2t &type;
};

typedef boost::shared_ptr<expr2t> expr2tc;

/** Constant class type. Not designed to contain any piece of data or method in
 *  particular, but allows for additional type safety. */
class constant2t : expr2t
{
protected:
  constant2t(type2t &type, expr_ids id);
  constant2t(const constant2t &ref);

public:
  /** Clone method. Entirely self explanatory */
  virtual expr2tc clone(void) const = 0;
};

/** Constant integer class. Records a constant integer of an arbitary
 *  precision */
class constant_int2t : constant2t
{
protected:
  constant_int2t(const BigInt &input);
  constant_int2t(const constant_int2t &ref);

  /** Concrete clone implementation. */
  virtual expr2tc clone(void) const;

  /** Arbitary precision integer record. */
  BigInt constant_value;
};

/** Constant class for string constants. */
class constant_string2t : constant2t
{
protected:
  constant_string2t(const std::string stringref &ref);
  constant_string2t(const constant_string2t &ref);

public:
  /** Concrete clone implementation. */
  virtual expr2tc clone(void) const;

  /** Arbitary precision integer record. */
  const std::string value;
};

/** Const datatype - for holding structs and unions */
class constant_datatype2t : constant2t
{
protected:
  constant_datatype2t(const type2t &type, const std::vector<exprt *> &members);
  constant_datatype2t(const constant_datatype2t &ref);

public:
  virtual expr2tc clone(void) const = 0;

  std::vector<exprt *> datatype_members;
};

class constant_struct2t : constant_datatype2t
{
protected:
  constant_struct2t(const type2t &type, const std::vector<exprt *> &members);
  constant_struct2t(const constant_struct2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class constant_union2t : constant_datatype2t
{
protected:
  constant_union2t(const type2t &type, const std::vector<exprt *> &members);
  constant_union2t(const constant_union2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class constant_array2t : constant2t
{
protected:
  constant_array2t(const type2t &type, const std::vector<exprt *> &members);
  constant_array2t(const constant_array2t &ref);

public:
  virtual expr2tc clone(void) const;

  std::vector<exprt *> datatype_members;
};

class constant_array_of2t : constant2t
{
protected:
  constant_array2t(const type2t &type, const expr2tc initializer);
  constant_array2t(const constant_array_of2t &ref);

public:
  virtual expr2tc clone(void) const;

  // Type records the size of the array; this records the initializer.
  const expr2tc initializer;
};

class symbol2t : expr2t
{
protected:
  symbol2t(const type2t &type, const std::string &name);
  symbol2t(const symbol2t &ref);

public:
  virtual expr2tc clone(void) const;

  // Symbol name - only so long as a symbol is a string. In the future, this
  // should really really change.
  const std::string name;
};

class typecast2t : expr2t
{
protected:
  typecast2t(const type2t &type, const expr2tc expr);
  typecast2t(const typecast2t &ref);

public:
  virtual expr2tc clone(void) const;

  // Expression to typecast from.
  const expr2tc from;
};

class if2t : expr2t
{
protected:
  if2t(const type2t &type, const expr2tc cond, const expr2tc true_val,
             const expr2tc false_val);
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
class rel2t : expr2t
{
protected:
  rel2t(const expr2tc val1, const expr2tc val2);
  rel2t(const rel2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc side_1;
  const expr2tc side_2;
};

class equality2t : rel2t
{
protected:
  equality2t(const expr2tc val1, const expr2tc val2);
  equality2t(const equality2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class notequal2t : rel2t
{
protected:
  notequal2t(const expr2tc val1, const expr2tc val2);
  notequal2t(const notequal2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class lessthan2t : rel2t
{
protected:
  lessthan2t(const expr2tc val1, const expr2tc val2);
  lessthan2t(const lessthan2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class greaterthan2t : rel2t
{
protected:
  greaterthan2t(const expr2tc val1, const expr2tc val2);
  greaterthan2t(const greaterthan2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class lessthanequal2t : rel2t
{
protected:
  lessthanequal2t(const expr2tc val1, const expr2tc val2);
  lessthanequal2t(const lessthanequal2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class greaterthanequal2t : rel2t
{
protected:
  greaterthanequal2t(const expr2tc val1, const expr2tc val2);
  greaterthanequal2t(const greaterthanequal2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Logical operations base class. Base for any logical operator. No storage in
 *  this particular class. Result is always of boolean type. */
class lops2t : expr2t
{
protected:
  lops2t(void);
  lops2t(const lops2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Not operator. Takes a boolean value; results in a boolean value. */
class not2t : lops2t
{
protected:
  not2t(const expr2tc notval);
  not2t(const not2t &ref);

public:
  virtual expr2tc clone(void) const;

  const exprt &notvalue;
};

/** Dynamic object operation. Checks to see whether or not the object is a
 *  dynamically allocated object or not. */
class dynamic_object2t : lops2t
{
protected:
  dynamic_object2t(const expr2tc val);
  dynamic_object2t(const dynamic_object2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc ptr_obj
};

/** Isnan operation. Checks whether expression is a NaN or not. */
class isnan2t : lops2t
{
protected:
  isnan2t(const expr2tc val);
  isnan2t(const isnan2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc value;
};

/** Base class for 2-operand boolean oeprators. Always results in a boolean,
 *  takes two operands, both of boolean type. */
class logical_2ops2t : lops2t
{
protected:
  logical_2ops2t(const expr2tc val1, const expr2tc val2);
  logical_2ops2t(const logical_2ops2t &ref);

public:
  virtual expr2tc clone(void) const;

  const exprt &side_1;
  const exprt &side_2;
};

class and2t : logical_2ops2t
{
protected:
  and2t(const expr2tc val1, const expr2tc val2);
  and2t(const and2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class or2t : logical_2ops2t
{
protected:
  or2t(const expr2tc val1, const expr2tc val2);
  or2t(const or2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class xor2t : logical_2ops2t
{
protected:
  xor2t(const expr2tc val1, const expr2tc val2);
  xor2t(const xor2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Same object operation. Compares two pointer objects to see if they're the
 *  same, with a boolean result. */
class same_object2t : logical_2ops2t
{
protected:
  same_object2t(const expr2tc val1, const expr2tc val2);
  same_object2t(const same_object2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Binary operations base class. Take a type, probably integer with a width,
 *  and some operands. */
class binops2t : expr2t
{
protected:
  binops2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  binops2t(const binops2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc side_1;
  const expr2tc side_2;
};

class bitand2t : binops2t
{
protected:
  bitand2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  bitand2t(const bitand2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class bitor2t : binops2t
{
protected:
  bitor2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  bitor2t(const bitor2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class bitxor2t : binops2t
{
protected:
  bitxor2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  bitxor2t(const bitxor2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class bitnand2t : binops2t
{
protected:
  bitnand2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  bitnand2t(const bitnand2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class bitnor2t : binops2t
{
protected:
  bitnor2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  bitnor2t(const bitnor2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class bitnxor2t : binops2t
{
protected:
  bitnxor2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  bitnxor2t(const bitnxor2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class lshr2t : binops2t
{
protected:
  lshr2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  lshr2t(const lshr2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Arithmatic base class. For all operations that are essentially integer
 *  arithmatic. */
class arith2t : expr2t
{
protected:
  arith2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  arith2t(const arith2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class neg2t : arith2t
{
protected:
  neg2t(const type2t &type, const expr2tc value);
  neg2t(const neg2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc value;
};

class abs2t : arith2t
{
protected:
  abs2t(const type2t &type, const expr2tc value);
  abs2t(const abs2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc value;
};

/** Base two-operand arithmatic class. */
class arith_2op2t : arith2t
{
protected:
  arith_2op2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  arith_2op2t(const arith_2op2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc part_1;
  const expr2tc part_2;
};

class add2t : arith_2op2t
{
protected:
  add2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  add2t(const add2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class sub2t : arith_2op2t
{
protected:
  sub2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  sub2t(const sub2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class mul2t : arith_2op2t
{
protected:
  mul2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  mul2t(const mul2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class div2t : arith_2op2t
{
protected:
  div2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  div2t(const div2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class modulus2t : arith_2op2t
{
protected:
  modulus2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  modulus2t(const modulus2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class shl2t : arith_2op2t
{
protected:
  shl2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  shl2t(const shl2t &ref);

public:
  virtual expr2tc clone(void) const;
};

class ashr2t : arith_2op2t
{
protected:
  ashr2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  ashr2t(const ashr2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Pointer offset. Extract pointer offset from a pointer value. Subclass of
 *  arithmatic because it returns an integer. */
class pointer_offset2t : arith2t
{
protected:
  pointer_offset2t(const expr2tc pointer);
  pointer_offset2t(const arith2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc pointer_obj;
};

/** Pointer object. Extract pointer object from a pointer value. Subclass of
 *  arithmatic because it returns an integer. */
class pointer_object2t : arith2t
{
protected:
  pointer_object2t(const expr2tc pointer);
  pointer_object2t(const pointer_object2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc pointer_obj;
};

/** Base class for byte operations. Endianness is a global property of the
 *  model that we're building, and we only need to care about it when we build
 *  an smt model in the end, not at any other point. */
class byte_ops2t : expr2t
{
protected:
  byte_ops2t(const type2t &type, const expr2tc val1, const expr2tc val2);
  byte_ops2t(const byte_ops2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** Data extraction from some expression. Type is whatever type we're expecting
 *  to pull out of it. source_value is whatever piece of data we're operating
 *  upon. source_offset is the _byte_ offset into source_value to extract data
 *  from. */
class byte_extract2t : byte_ops2t
{
protected:
  byte_extract2t(const type2t &type, const expr2tc source, const expr2tc offs);
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
class byte_update2t : byte_ops2t
{
protected:
  byte_udpate2t(const type2t &type, const expr2tc source, const expr2tc offs,
                const expr2tc update);
  byte_update2t(const byte_update2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc source_value;
  const expr2tc source_offset;
  const expr2tc update_value;
};

/** Base type of datatype operations. */
class datatype_ops2t : expr2t
{
protected:
  datatype_ops2t(const type2t &type);
  datatype_ops2t(const datatype_ops2t &ref);

public:
  virtual expr2tc clone(void) const;
};

/** With operation. Some kind of piece of data, another piece of data to
 *  insert into it, and where to put it. */
class with2t : datatype_ops2t
{
protected:
  with2t(const type2t &type, const expr2tc source, const expr2tc update,
         const int field);
  with2t(const with2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc source_data;
  const expr2tc update_data;
  const int update_field;
};

/** Member operation. Extracts some field from a datatype. */
class member2t : datatype_ops2t
{
protected:
  member2t(const type2t &type, const expr2tc source, const int field);
  member2t(const member2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc source_data;
  const int field;
};

/** Index operation. Extracts an entry from an array. */
class index2t : datatype_ops2t
{
protected:
  index2t(const type2t &type, const expr2tc source, const expr2tc index);
  index2t(const index2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc source_data;
  const expr2tc index;
};

/** Zero string operation. Don't quite understand it. Just operates on the
 *  string struct as far as I know. Result is boolean. */
class zero_string2t : datatype_ops2t
{
protected:
  zero_string2t(const expr2tc string);
  zero_string2t(const with2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc string;
};

/** Zero length string. Unknown dirference from zero_string. */
class zero_length_string2t : datatype_ops2t
{
protected:
  zero_length_string2t(const expr2tc string);
  zero_length_string2t(const with2t &ref);

public:
  virtual expr2tc clone(void) const;

  const expr2tc string;
};
