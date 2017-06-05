/*
 * irep2_type.h
 *
 *  Created on: Jun 2, 2017
 *      Author: mramalho
 */

#ifndef IREP2_TYPE_H_
#define IREP2_TYPE_H_

#include <util/irep2.h>

// Start with forward class definitions

class bool_type2t;
class empty_type2t;
class symbol_type2t;
class struct_type2t;
class union_type2t;
class bv_type2t;
class unsignedbv_type2t;
class signedbv_type2t;
class code_type2t;
class array_type2t;
class pointer_type2t;
class fixedbv_type2t;
class floatbv_type2t;
class string_type2t;
class cpp_name_type2t;

// We also require in advance, the actual classes that store type data.

class symbol_type_data : public type2t
{
public:
  symbol_type_data(type2t::type_ids id, const dstring sym_name) :
    type2t (id), symbol_name(sym_name) {}
  symbol_type_data(const symbol_type_data &ref) :
    type2t (ref), symbol_name(ref.symbol_name) { }

  irep_idt symbol_name;

// Type mangling:
  typedef esbmct::field_traits<irep_idt, symbol_type_data, &symbol_type_data::symbol_name> symbol_name_field;
  typedef esbmct::type2t_traits<symbol_name_field> traits;
};

class struct_union_data : public type2t
{
public:
  struct_union_data(type2t::type_ids id, const std::vector<type2tc> &membs,
    const std::vector<irep_idt> &names, const std::vector<irep_idt> &pretty_names,
    const irep_idt &n)
      : type2t(id), members(membs), member_names(names),
        member_pretty_names(pretty_names), name(n)
  {
  }
  struct_union_data(const struct_union_data &ref)
    : type2t(ref), members(ref.members), member_names(ref.member_names),
      member_pretty_names(ref.member_pretty_names), name(ref.name) { }

  /** Fetch index number of member. Given a textual name of a member of a
   *  struct or union, this method will look up what index it is into the
   *  vector of types that make up this struct/union. Always returns the correct
   *  index, if you give it a name that isn't part of this struct/union it'll
   *  abort.
   *  @param name Name of member of this struct/union to look up.
   *  @return Index into members/member_names vectors */
  unsigned int get_component_number(const irep_idt &name) const;

  const std::vector<type2tc> & get_structure_members(void) const;
  const std::vector<irep_idt> & get_structure_member_names(void) const;
  const irep_idt & get_structure_name(void) const;

  std::vector<type2tc> members;
  std::vector<irep_idt> member_names;
  std::vector<irep_idt> member_pretty_names;
  irep_idt name;

// Type mangling:
  typedef esbmct::field_traits<std::vector<type2tc>, struct_union_data, &struct_union_data::members> members_field;
  typedef esbmct::field_traits<std::vector<irep_idt>, struct_union_data, &struct_union_data::member_names> member_names_field;
  typedef esbmct::field_traits<std::vector<irep_idt>, struct_union_data, &struct_union_data::member_pretty_names> member_pretty_names_field;
  typedef esbmct::field_traits<irep_idt, struct_union_data, &struct_union_data::name> name_field;
  typedef esbmct::type2t_traits<members_field, member_names_field, member_pretty_names_field, name_field> traits;
};

class bv_data : public type2t
{
public:
  bv_data(type2t::type_ids id, unsigned int w) : type2t(id), width(w)
  {
    // assert(w != 0 && "Must have nonzero width for integer type");
    // XXX -- zero sized bitfields are permissible. Oh my.
  }
  bv_data(const bv_data &ref) : type2t(ref), width(ref.width) { }

  virtual unsigned int get_width(void) const;

  unsigned int width;

// Type mangling:
  typedef esbmct::field_traits<unsigned int, bv_data, &bv_data::width> width_field;
  typedef esbmct::type2t_traits<width_field> traits;
};

class code_data : public type2t
{
public:
  code_data(type2t::type_ids id, const std::vector<type2tc> &args,
            const type2tc &ret, const std::vector<irep_idt> &names, bool e)
    : type2t(id), arguments(args), ret_type(ret), argument_names(names),
      ellipsis(e) { }
  code_data(const code_data &ref)
    : type2t(ref), arguments(ref.arguments), ret_type(ref.ret_type),
      argument_names(ref.argument_names), ellipsis(ref.ellipsis) { }

  virtual unsigned int get_width(void) const;

  std::vector<type2tc> arguments;
  type2tc ret_type;
  std::vector<irep_idt> argument_names;
  bool ellipsis;

// Type mangling:
  typedef esbmct::field_traits<std::vector<type2tc>, code_data, &code_data::arguments> arguments_field;
  typedef esbmct::field_traits<type2tc, code_data, &code_data::ret_type> ret_type_field;
  typedef esbmct::field_traits<std::vector<irep_idt>, code_data, &code_data::argument_names> argument_names_field;
  typedef esbmct::field_traits<bool, code_data, &code_data::ellipsis> ellipsis_field;
  typedef esbmct::type2t_traits<arguments_field, ret_type_field, argument_names_field, ellipsis_field> traits;
};

class array_data : public type2t
{
public:
  array_data(type2t::type_ids id, const type2tc &st, const expr2tc &sz, bool i)
    : type2t(id), subtype(st), array_size(sz), size_is_infinite(i) { }
  array_data(const array_data &ref)
    : type2t(ref), subtype(ref.subtype), array_size(ref.array_size),
      size_is_infinite(ref.size_is_infinite) { }

  type2tc subtype;
  expr2tc array_size;
  bool size_is_infinite;

// Type mangling:
  typedef esbmct::field_traits<type2tc, array_data, &array_data::subtype> subtype_field;
  typedef esbmct::field_traits<expr2tc, array_data, &array_data::array_size> array_size_field;
  typedef esbmct::field_traits<bool, array_data, &array_data::size_is_infinite> size_is_infinite_field;
  typedef esbmct::type2t_traits<subtype_field, array_size_field, size_is_infinite_field> traits;
};

class pointer_data : public type2t
{
public:
  pointer_data(type2t::type_ids id, const type2tc &st)
    : type2t(id), subtype(st) { }
  pointer_data(const pointer_data &ref)
    : type2t(ref), subtype(ref.subtype) { }

  type2tc subtype;

// Type mangling:
  typedef esbmct::field_traits<type2tc, pointer_data, &pointer_data::subtype> subtype_field;
  typedef esbmct::type2t_traits<subtype_field> traits;
};

class fixedbv_data : public type2t
{
public:
  fixedbv_data(type2t::type_ids id, unsigned int w, unsigned int ib)
    : type2t(id), width(w), integer_bits(ib) { }
  fixedbv_data(const fixedbv_data &ref)
    : type2t(ref), width(ref.width), integer_bits(ref.integer_bits) { }

  unsigned int width;
  unsigned int integer_bits;

// Type mangling:
  typedef esbmct::field_traits<unsigned int, fixedbv_data, &fixedbv_data::width> width_field;
  typedef esbmct::field_traits<unsigned int, fixedbv_data, &fixedbv_data::integer_bits> integer_bits_field;
  typedef esbmct::type2t_traits<width_field, integer_bits_field> traits;
};

class floatbv_data : public type2t
{
public:
  floatbv_data(type2t::type_ids id, unsigned int f, unsigned int e)
    : type2t(id), fraction(f), exponent(e) { }
  floatbv_data(const floatbv_data &ref)
    : type2t(ref), fraction(ref.fraction), exponent(ref.exponent) { }

  unsigned int fraction;
  unsigned int exponent;

// Type mangling:
  typedef esbmct::field_traits<unsigned int, floatbv_data, &floatbv_data::fraction> fraction_field;
  typedef esbmct::field_traits<unsigned int, floatbv_data, &floatbv_data::exponent> exponent_field;
  typedef esbmct::type2t_traits<fraction_field, exponent_field> traits;
};

class string_data : public type2t
{
public:
  string_data(type2t::type_ids id, unsigned int w)
    : type2t(id), width(w) { }
  string_data(const string_data &ref)
    : type2t(ref), width(ref.width) { }

  unsigned int width;

// Type mangling:
  typedef esbmct::field_traits<unsigned int, string_data, &string_data::width> width_field;
  typedef esbmct::type2t_traits<width_field> traits;
};

class cpp_name_data : public type2t
{
public:
  cpp_name_data(type2t::type_ids id, const irep_idt &n,
                const std::vector<type2tc> &templ_args)
    : type2t(id), name(n), template_args(templ_args) { }
  cpp_name_data(const cpp_name_data &ref)
    : type2t(ref), name(ref.name), template_args(ref.template_args) { }

  irep_idt name;
  std::vector<type2tc> template_args;

// Type mangling:
  typedef esbmct::field_traits<irep_idt, cpp_name_data, &cpp_name_data::name> name_field;
  typedef esbmct::field_traits<std::vector<type2tc>, cpp_name_data, &cpp_name_data::template_args> template_args_field;
  typedef esbmct::type2t_traits<name_field, template_args_field> traits;
};

// Then give them a typedef name

#define irep_typedefs(basename, superclass) \
  typedef esbmct::something2tc<type2t, basename##_type2t,\
                              type2t::basename##_id, const type2t::type_ids,\
                              &type2t::type_id, superclass> basename##_type2tc;\
  typedef esbmct::type_methods2<basename##_type2t, superclass, superclass::traits, basename##_type2tc> basename##_type_methods;\
  extern template class esbmct::type_methods2<basename##_type2t, superclass, superclass::traits, basename##_type2tc>;

irep_typedefs(bool, type2t)
irep_typedefs(empty, type2t)
irep_typedefs(symbol, symbol_type_data)
irep_typedefs(struct, struct_union_data)
irep_typedefs(union, struct_union_data)
irep_typedefs(unsignedbv, bv_data)
irep_typedefs(signedbv, bv_data)
irep_typedefs(code, code_data)
irep_typedefs(array, array_data)
irep_typedefs(pointer, pointer_data)
irep_typedefs(fixedbv, fixedbv_data)
irep_typedefs(floatbv, floatbv_data)
irep_typedefs(string, string_data)
irep_typedefs(cpp_name, cpp_name_data)
#undef irep_typedefs

/** Boolean type.
 *  Identifies a boolean type. Contains no additional data.
 *  @extends typet
 */
class bool_type2t : public bool_type_methods
{
public:
  bool_type2t(void) : bool_type_methods (bool_id) {}
  bool_type2t(const bool_type2t &ref) : bool_type_methods(ref) {}
  virtual unsigned int get_width(void) const;

  static std::string field_names[esbmct::num_type_fields];
};

/** Empty type.
 *  For void pointers and the like, with no type. No extra data.
 *  @extends type2t
 */
class empty_type2t : public empty_type_methods
{
public:
  empty_type2t(void) : empty_type_methods(empty_id) {}
  empty_type2t(const empty_type2t &ref) : empty_type_methods(ref) { }
  virtual unsigned int get_width(void) const;

  static std::string field_names[esbmct::num_type_fields];
};

/** Symbolic type.
 *  Temporary, prior to linking up types after parsing, or when a struct/array
 *  contains a recursive pointer to its own type.
 *  @extends symbol_type_data
 */
class symbol_type2t : public symbol_type_methods
{
public:
  /** Primary constructor. @param sym_name Name of symbolic type. */
  symbol_type2t(const dstring &sym_name) :
    symbol_type_methods(symbol_id, sym_name) { }
  symbol_type2t(const symbol_type2t &ref) :
    symbol_type_methods(ref) { }
  virtual unsigned int get_width(void) const;

  static std::string field_names[esbmct::num_type_fields];
};

/** Struct type.
 *  Represents both C structs and the data in C++ classes. Contains a vector
 *  of types recording what type each member is, a vector of names recording
 *  what the member names are, and a name for the struct.
 *  @extends struct_union_data
 */
class struct_type2t : public struct_type_methods
{
public:
  /** Primary constructor.
   *  @param members Vector of types for the members in this struct.
   *  @param memb_names Vector of names for the members in this struct.
   *  @param name Name of this struct.
   */
  struct_type2t(const std::vector<type2tc> &members,
                const std::vector<irep_idt> &memb_names,
                const std::vector<irep_idt> &memb_pretty_names,
                const irep_idt &name)
    : struct_type_methods(struct_id, members, memb_names, memb_pretty_names, name) {}
  struct_type2t(const struct_type2t &ref) : struct_type_methods(ref) {}
  virtual unsigned int get_width(void) const;

  static std::string field_names[esbmct::num_type_fields];
};

/** Union type.
 *  Represents a union type - in a similar vein to struct_type2t, this contains
 *  a vector of types and vector of names, each element of which corresponds to
 *  a member in the union. There's also a name for the union.
 *  @extends struct_union_data
 */
class union_type2t : public union_type_methods
{
public:
  /** Primary constructor.
   *  @param members Vector of types corresponding to each member of union.
   *  @param memb_names Vector of names corresponding to each member of union.
   *  @param name Name of this union
   */
  union_type2t(const std::vector<type2tc> &members,
               const std::vector<irep_idt> &memb_names,
               const std::vector<irep_idt> &memb_pretty_names,
               const irep_idt &name)
    : union_type_methods(union_id, members, memb_names, memb_pretty_names, name) {}
  union_type2t(const union_type2t &ref) : union_type_methods(ref) {}
  virtual unsigned int get_width(void) const;

  static std::string field_names[esbmct::num_type_fields];
};

/** Unsigned integer type.
 *  Represents any form of unsigned integer; the size of this integer is
 *  recorded in the width field.
 *  @extends bv_data
 */
class unsignedbv_type2t : public unsignedbv_type_methods
{
public:
  /** Primary constructor. @param width Width of represented integer */
  unsignedbv_type2t(unsigned int width)
    : unsignedbv_type_methods(unsignedbv_id, width) { }
  unsignedbv_type2t(const unsignedbv_type2t &ref)
    : unsignedbv_type_methods(ref) { }

  static std::string field_names[esbmct::num_type_fields];
};

/** Signed integer type.
 *  Represents any form of signed integer; the size of this integer is
 *  recorded in the width field.
 *  @extends bv_data
 */
class signedbv_type2t : public signedbv_type_methods
{
public:
  /** Primary constructor. @param width Width of represented integer */
  signedbv_type2t(signed int width)
    : signedbv_type_methods(signedbv_id, width) { }
  signedbv_type2t(const signedbv_type2t &ref)
    : signedbv_type_methods(ref) { }

  static std::string field_names[esbmct::num_type_fields];
};

/** Empty type. For void pointers and the like, with no type. No extra data */
class code_type2t : public code_type_methods
{
public:
  code_type2t(const std::vector<type2tc> &args, const type2tc &ret_type,
              const std::vector<irep_idt> &names, bool e)
    : code_type_methods(code_id, args, ret_type, names, e)
  { assert(args.size() == names.size()); }
  code_type2t(const code_type2t &ref) : code_type_methods(ref) { }

  static std::string field_names[esbmct::num_type_fields];
};

/** Array type.
 *  Comes with a subtype of the array and a size that might be constant, might
 *  be nondeterministic, might be infinite. These facts are recorded in the
 *  array_size and size_is_infinite fields.
 *
 *  If size_is_infinite is true, array_size will be null. If array_size is
 *  not a constant number, then it's a dynamically sized array.
 *  @extends array_data
 */
class array_type2t : public array_type_methods
{
public:
  /** Primary constructor.
   *  @param subtype Type of elements in this array.
   *  @param size Size of this array.
   *  @param inf Whether or not this array is infinitely sized
   */
  array_type2t(const type2tc &_subtype, const expr2tc &size, bool inf)
    : array_type_methods (array_id, _subtype, size, inf) {
      // If we can simplify the array size, do so
      // XXX, this is probably massively inefficient. Some kind of boundry in
      // the checking process should exist to eliminate this requirement.
      if (!is_nil_expr(size)) {
        expr2tc sz = size->simplify();
        if (!is_nil_expr(sz))
          array_size = sz;
      }
    }
  array_type2t(const array_type2t &ref)
    : array_type_methods(ref) { }

  virtual unsigned int get_width(void) const;

  /** Exception for invalid manipulations of an infinitely sized array. No
   *  actual data stored. */
  class inf_sized_array_excp {
  };

  /** Exception for invalid manipultions of dynamically sized arrays.
   *  Stores the size of the array in the exception; this way the catcher
   *  has it immediately to hand. */
  class dyn_sized_array_excp {
  public:
    dyn_sized_array_excp(const expr2tc _size) : size(_size) {}
    expr2tc size;
  };

  static std::string field_names[esbmct::num_type_fields];
};

/** Pointer type.
 *  Simply has a subtype, of what it points to. No other attributes.
 *  @extends pointer_data
 */
class pointer_type2t : public pointer_type_methods
{
public:
  /** Primary constructor. @param subtype Subtype of this pointer */
  pointer_type2t(const type2tc &subtype)
    : pointer_type_methods(pointer_id, subtype) { }
  pointer_type2t(const pointer_type2t &ref)
    : pointer_type_methods(ref) { }
  virtual unsigned int get_width(void) const;

  static std::string field_names[esbmct::num_type_fields];
};

/** Fixed bitvector type.
 *  Contains a spec for a fixed bitwidth number -- this is the equivalent of a
 *  fixedbv_spect in the old irep situation. Stores how bits are distributed
 *  over integer bits and fraction bits.
 *  @extend fixedbv_data
 */
class fixedbv_type2t : public fixedbv_type_methods
{
public:
  /** Primary constructor.
   *  @param width Total number of bits in this type of fixedbv
   *  @param integer Number of integer bits in this type of fixedbv
   */
  fixedbv_type2t(unsigned int width, unsigned int integer)
    : fixedbv_type_methods(fixedbv_id, width, integer) { }
  fixedbv_type2t(const fixedbv_type2t &ref)
    : fixedbv_type_methods(ref) { }
  virtual unsigned int get_width(void) const;

  static std::string field_names[esbmct::num_type_fields];
};

/** Floating-point bitvector type.
 *  Contains a spec for a floating point number -- this is the equivalent of a
 *  ieee_float_spect in the old irep situation. Stores how bits are distributed
 *  over fraction bits and exponent bits.
 *  @extend floatbv_type_methods
 */
class floatbv_type2t : public floatbv_type_methods
{
public:
  /** Primary constructor.
   *  @param fraction Number of fraction bits in this type of floatbv
   *  @param exponent Number of exponent bits in this type of floatbv
   */
  floatbv_type2t(unsigned int fraction, unsigned int exponent)
    : floatbv_type_methods(floatbv_id, fraction, exponent) { }
  floatbv_type2t(const floatbv_type2t &ref)
    : floatbv_type_methods(ref) { }
  virtual unsigned int get_width(void) const;

  static std::string field_names[esbmct::num_type_fields];
};

/** String type class.
 *  Slightly artificial as original irep had no type for this; Represents the
 *  type of a string constant. Because it needs a bit width, we also store the
 *  size of the constant string in elements.
 *  @extends string_data
 */
class string_type2t : public string_type_methods
{
public:
  /** Primary constructor.
   *  @param elements Number of 8-bit characters in string constant.
   */
  string_type2t(unsigned int elements)
    : string_type_methods(string_id, elements) { }
  string_type2t(const string_type2t &ref)
    : string_type_methods(ref) { }
  virtual unsigned int get_width(void) const;
  virtual unsigned int get_length(void) const;

  static std::string field_names[esbmct::num_type_fields];
};

/** C++ Name type.
 *  Contains a type name, but also a vector of template parameters.
 *  Something in the C++ frontend uses this; it's precise purpose is unclear.
 *  @extends cpp_name_data
 */
class cpp_name_type2t : public cpp_name_type_methods
{
public:
  /** Primary constructor.
   *  @param n Name of this type.
   *  @param ta Vector of template arguments (types).
   */
  cpp_name_type2t(const irep_idt &n, const std::vector<type2tc> &ta)
    : cpp_name_type_methods(cpp_name_id, n, ta){}
  cpp_name_type2t(const cpp_name_type2t &ref)
    : cpp_name_type_methods(ref) { }

  virtual unsigned int get_width(void) const;

  static std::string field_names[esbmct::num_type_fields];
};

// Generate some "is-this-a-blah" macros, and type conversion macros. This is
// fine in terms of using/ keywords in syntax, because the preprocessor
// preprocesses everything out.
#define type_macros(name) \
  inline bool is_##name##_type(const expr2tc &e) \
    { return e->type->type_id == type2t::name##_id; } \
  inline bool is_##name##_type(const type2tc &t) \
    { return t->type_id == type2t::name##_id; } \
  inline const name##_type2t & to_##name##_type(const type2tc &t) \
    { return dynamic_cast<const name##_type2t &> (*t.get()); } \
  inline name##_type2t & to_##name##_type(type2tc &t) \
    { return dynamic_cast<name##_type2t &> (*t.get()); } \
  inline name##_type2t & to_##name##_type(type2t &t) \
     { return dynamic_cast<name##_type2t &> (t); } \
  inline const name##_type2t & to_##name##_type(const type2t &t) \
     { return dynamic_cast<const name##_type2t &> (t); }

type_macros(bool);
type_macros(empty);
type_macros(symbol);
type_macros(struct);
type_macros(union);
type_macros(code);
type_macros(array);
type_macros(pointer);
type_macros(unsignedbv);
type_macros(signedbv);
type_macros(fixedbv);
type_macros(floatbv);
type_macros(string);
type_macros(cpp_name);
#undef type_macros
#ifdef dynamic_cast
#undef dynamic_cast
#endif

/** Pool for caching converted types.
 *  Various common types (bool, empty for example) needn't be reallocated
 *  every time we need a new one; it's better to have some global constants
 *  of them, which is what this class provides. There are global bool and empty
 *  types to be used; in addition, there are helper methods to create integer
 *  types with common bit widths, and methods to enter a used type into a cache
 *  of them, allowing migration of typet <=> type2t to be faster.
 */
class type_poolt {
public:
  type_poolt(void);
  type_poolt(bool yolo);

  type_poolt &operator=(type_poolt const &ref);

  type2tc bool_type;
  type2tc empty_type;

  const type2tc &get_bool() const { return bool_type; }
  const type2tc &get_empty() const { return empty_type; }

  // For other types, have a pool of them for quick lookup.
  std::map<typet, type2tc> struct_map;
  std::map<typet, type2tc> union_map;
  std::map<typet, type2tc> array_map;
  std::map<typet, type2tc> pointer_map;
  std::map<typet, type2tc> unsignedbv_map;
  std::map<typet, type2tc> signedbv_map;
  std::map<typet, type2tc> fixedbv_map;
  std::map<typet, type2tc> floatbv_map;
  std::map<typet, type2tc> string_map;
  std::map<typet, type2tc> symbol_map;
  std::map<typet, type2tc> code_map;

  // And refs to some of those for /really/ quick lookup;
  const type2tc *uint8;
  const type2tc *uint16;
  const type2tc *uint32;
  const type2tc *uint64;
  const type2tc *int8;
  const type2tc *int16;
  const type2tc *int32;
  const type2tc *int64;

  // Some accessors.
  const type2tc &get_struct(const typet &val);
  const type2tc &get_union(const typet &val);
  const type2tc &get_array(const typet &val);
  const type2tc &get_pointer(const typet &val);
  const type2tc &get_unsignedbv(const typet &val);
  const type2tc &get_signedbv(const typet &val);
  const type2tc &get_fixedbv(const typet &val);
  const type2tc &get_floatbv(const typet &val);
  const type2tc &get_string(const typet &val);
  const type2tc &get_symbol(const typet &val);
  const type2tc &get_code(const typet &val);

  const type2tc &get_uint(unsigned int size);
  const type2tc &get_int(unsigned int size);

  const type2tc &get_uint8() const { return *uint8; }
  const type2tc &get_uint16() const { return *uint16; }
  const type2tc &get_uint32() const { return *uint32; }
  const type2tc &get_uint64() const { return *uint64; }
  const type2tc &get_int8() const { return *int8; }
  const type2tc &get_int16() const { return *int16; }
  const type2tc &get_int32() const { return *int32; }
  const type2tc &get_int64() const { return *int64; }
};

extern type_poolt type_pool;

#endif /* IREP2_TYPE_H_ */
