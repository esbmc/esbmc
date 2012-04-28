#ifndef _UTIL_IREP2_H_
#define _UTIL_IREP2_H_

#include <stdarg.h>

#include <vector>

#include <boost/mpl/if.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/crc.hpp>

#include <irep.h>
#include <fixedbv.h>
#include <big-int/bigint.hh>
#include <dstring.h>

// XXXjmorse - abstract, access modifies, need consideration

#define forall_exprs(it, vect) \
  for (std::vector<expr2tc>::const_iterator (it) = (vect).begin();\
       it != (vect).end(); it++)

#define Forall_exprs(it, vect) \
  for (std::vector<expr2tc>::iterator (it) = (vect).begin();\
       it != (vect).end(); it++

#define forall_types(it, vect) \
  for (std::vector<type2tc>::const_iterator (it) = (vect).begin();\
       it != (vect).end(); it++)

#define Forall_types(it, vect) \
  for (std::vector<type2tc>::iterator (it) = (vect).begin();\
       it != (vect).end(); it++)

#define forall_names(it, vect) \
  for (std::vector<std::string>::const_iterator (it) = (vect).begin();\
       it != (vect).end(); it++)

#define Forall_names(it, vect) \
  for (std::vector<std::string>::iterator (it) = (vect).begin();\
       it != (vect).end(); it++)

class prop_convt;

class type2t;
class expr2t;
class constant_array2t;
typedef boost::shared_ptr<type2t> type2tc;
typedef boost::shared_ptr<expr2t> expr2tc;

typedef std::pair<std::string,std::string> member_entryt;
typedef std::vector<member_entryt> list_of_memberst;

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
    string_id,
    end_type_id
  };

  // Class to be thrown when attempting to fetch the width of a symbolic type,
  // such as empty or code. Caller will have to worry about what to do about
  // that.
  class symbolic_type_excp {
  };

protected:
  type2t(type_ids id);
  type2t(const type2t &ref);

public:
  virtual void convert_smt_type(const prop_convt &obj, void *&arg) const = 0;
  virtual unsigned int get_width(void) const = 0;
  bool operator==(const type2t &ref) const;
  bool operator!=(const type2t &ref) const;
  bool operator<(const type2t &ref) const;
  int ltchecked(const type2t &ref) const;
  std::string pretty(unsigned int indent = 0) const;
  void dump(void) const;
  uint32_t crc(void) const;
  virtual bool cmp(const type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const = 0;
  virtual void do_crc(boost::crc_32_type &crc) const;

  /** Instance of type_ids recording this types type. */
  type_ids type_id;

  static const char *type_names[];
};

template <class derived>
class type_body : public type2t
{
protected:
  type_body(type_ids id) : type2t(id) {};
  type_body(const type_body &ref);

public:
  virtual void convert_smt_type(const prop_convt &obj, void *&arg) const;
};

/** Boolean type. No additional data */
class bool_type2t : public type_body<bool_type2t>
{
public:
  bool_type2t(void);
  virtual bool cmp(const bool_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual unsigned int get_width(void) const;
protected:
  bool_type2t(const bool_type2t &ref);
};

/** Empty type. For void pointers and the like, with no type. No extra data */
class empty_type2t : public type_body<empty_type2t>
{
public:
  empty_type2t(void);
  virtual bool cmp(const empty_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
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
  virtual bool cmp(const symbol_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;
  virtual unsigned int get_width(void) const;
protected:
  symbol_type2t(const symbol_type2t &ref);

public:
  const dstring symbol_name;
};

class struct_union_type2t : public type_body<struct_union_type2t>
{
protected:
  struct_union_type2t(type_ids id, const std::vector<type2tc> &members,
                      std::vector<std::string> memb_names, std::string name);
  struct_union_type2t(const struct_union_type2t &ref);

public:
  virtual bool cmp(const struct_union_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;
  const std::vector<type2tc> members;
  std::vector<std::string> member_names;
  std::string name;
};

class bv_type2t : public type_body<bv_type2t>
{
protected:
  bv_type2t(type2t::type_ids id, unsigned int width);
  bv_type2t(const bv_type2t &ref);

public:
  virtual  bool cmp(const bv_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;
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
  struct_union_type_body2t(type_ids id, const std::vector<type2tc> &members,
                           std::vector<std::string> memb_names, std::string name)
    : struct_union_type2t(id, members, memb_names, name) {};
  struct_union_type_body2t(const struct_union_type_body2t &ref)
    : struct_union_type2t(ref) {};

public:
  virtual void convert_smt_type(const prop_convt &obj, void *&arg) const;
};

template <class derived>
class bv_type_body : public bv_type2t
{
protected:
  bv_type_body(type_ids id, unsigned int width) : bv_type2t(id, width) {};
  bv_type_body(const bv_type_body &ref) : bv_type2t(ref) {};

public:
  virtual void convert_smt_type(const prop_convt &obj, void *&arg) const;
};

class struct_type2t : public struct_union_type_body2t<struct_type2t>
{
public:
  struct_type2t(std::vector<type2tc> &members,
                std::vector<std::string> memb_names,
                std::string name);
  virtual bool cmp(const struct_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual unsigned int get_width(void) const;
protected:
  struct_type2t(const struct_type2t &ref);
};

class union_type2t : public struct_union_type_body2t<union_type2t>
{
public:
  union_type2t(std::vector<type2tc> &members,
               std::vector<std::string> memb_names,
               std::string name);
  virtual bool cmp(const union_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual unsigned int get_width(void) const;
protected:
  union_type2t(const union_type2t &ref);
};

/** Code type. No additional data whatsoever. */
class code_type2t : public type_body<code_type2t>
{
public:
  code_type2t(void);
  virtual bool cmp(const code_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
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
  virtual bool cmp(const array_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;
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
  public:
    dyn_sized_array_excp(const expr2tc _size) : size(_size) {}
    expr2tc size;
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
  virtual bool cmp(const pointer_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;
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
  virtual bool cmp(const unsignedbv_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
protected:
  unsignedbv_type2t(const unsignedbv_type2t &ref);
};

class signedbv_type2t : public bv_type_body<signedbv_type2t>
{
public:
  signedbv_type2t(unsigned int width);
  virtual bool cmp(const signedbv_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
protected:
  signedbv_type2t(const signedbv_type2t &ref);
};

class fixedbv_type2t : public type_body<fixedbv_type2t>
{
public:
  fixedbv_type2t(unsigned int width, unsigned int integer);
  virtual bool cmp(const fixedbv_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;
  virtual unsigned int get_width(void) const;
protected:
  fixedbv_type2t(const fixedbv_type2t &ref);

public:
  const unsigned int width;
  const unsigned int integer_bits;
};

class string_type2t : public type_body<string_type2t>
{
public:
  string_type2t(unsigned int elements);
  virtual bool cmp(const string_type2t &ref) const;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;
  virtual unsigned int get_width(void) const;
  string_type2t(const string_type2t &ref);

  unsigned int elements;
};

// Generate some "is-this-a-blah" macros, and type conversion macros. This is
// fine in terms of using/ keywords in syntax, because the preprocessor
// preprocesses everything out. One more used to C++ templates might raise their
// eyebrows at using the preprocessor; nuts to you, this works.
#ifdef NDEBUG
#define dynamic_cast static_cast
#endif
#define type_macros(name) \
  inline bool is_##name##_type(const type2tc &t) \
    { return t->type_id == type2t::name##_id; } \
  inline const name##_type2t & to_##name##_type(const type2tc &t) \
    { return dynamic_cast<const name##_type2t &> (*t.get()); } \
  inline name##_type2t & to_##name##_type(type2tc &t) \
    { return dynamic_cast<name##_type2t &> (*t.get()); }

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
type_macros(string);
#undef type_macros
#ifdef dynamic_cast
#undef dynamic_cast
#endif

inline bool is_bv_type(const type2tc &t) \
{ return (t->type_id == type2t::unsignedbv_id ||
          t->type_id == type2t::signedbv_id); }

inline bool is_structure_type(const type2tc &t) \
{ return (t->type_id == type2t::union_id ||
          t->type_id == type2t::struct_id); }

inline const struct_union_type2t & to_structure_type(const type2tc &t)
  { return dynamic_cast<const struct_union_type2t &> (*t.get()); }
inline struct_union_type2t & to_structure_type(type2tc &t)
  { return dynamic_cast<struct_union_type2t &> (*t.get()); }

inline const bv_type2t & to_bv_type(const type2tc &t)
  { return dynamic_cast<const bv_type2t &> (*t.get()); }
inline bv_type2t & to_bv_type(type2tc &t)
  { return dynamic_cast<bv_type2t &> (*t.get()); }

/** Base class for all expressions */
class expr2t
{
public:
  /** Enumeration identifying each sort of expr.
   *  The idea being to permit runtime identification of a type for debugging or
   *  otherwise. See type2t::type_ids. */
  enum expr_ids {
    constant_int_id,
    constant_fixedbv_id,
    constant_bool_id,
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
    and_id,
    or_id,
    xor_id,
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
    dynamic_object_id, // Not converted in Z3, only in goto-symex
    same_object_id,
    pointer_offset_id,
    pointer_object_id,
    address_of_id,
    byte_extract_id,
    byte_update_id,
    with_id,
    member_id,
    index_id,
    zero_string_id,
    zero_length_string_id,
    isnan_id,
    overflow_id,
    overflow_cast_id,
    overflow_neg_id,
    end_expr_id
  };

  // Template metaprogramming (vomit) -- define tag classes to instanciate
  // different class fields with different names and different types.

  #define field_name_macro(name) \
  template <class fieldtype> \
  struct name_class_##name { \
  public: \
    fieldtype name; \
  }; \
  template <class fieldtype> \
  struct name_##name { \
  public: \
    typedef name_class_##name<fieldtype> type; \
  };

  field_name_macro(value);
  field_name_macro(datatype_members);
  field_name_macro(name);
  field_name_macro(from);
  field_name_macro(cond);
  field_name_macro(true_value);
  field_name_macro(false_value);
  field_name_macro(side_1);
  field_name_macro(side_2);
  field_name_macro(notvalue);
  field_name_macro(ptr_obj);
  field_name_macro(big_endian);
  field_name_macro(source_value);
  field_name_macro(source_offset);
  field_name_macro(update_value);
  field_name_macro(update_field);
  field_name_macro(member);
  field_name_macro(index);
  field_name_macro(string);
  field_name_macro(bits);
  #undef field_name_macro

  struct name_class_empty {
  public:
  };
  struct name_empty {
  public:
    typedef name_class_empty type;
  };

protected:
  expr2t(const type2tc type, expr_ids id);
  expr2t(const expr2t &ref);

public:
  /** Clone method. Entirely self explanatory */
  virtual expr2tc clone(void) const = 0;

  virtual void convert_smt(prop_convt &obj, void *&arg) const = 0;

  bool operator==(const expr2t &ref) const;
  bool operator<(const expr2t &ref) const;
  bool operator!=(const expr2t &ref) const;
  int ltchecked(const expr2t &ref) const;
  std::string pretty(unsigned int indent = 0) const;
  void dump(void) const;
  uint32_t crc(void) const;
  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const = 0;
  virtual void do_crc(boost::crc_32_type &crc) const;

  /** Instance of expr_ids recording tihs exprs type. */
  expr_ids expr_id;

  /** Type of this expr. All exprs have a type. */
  const type2tc type;

  static const char *expr_names[];
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

template <class derived, bool arg1, typename type1>
class expr_body2 : public expr2t, boost::mpl::if_c<arg1, typename type1::type, expr2t::name_empty>::type
{
protected:
  expr_body2(const type2tc type, expr_ids id) : expr2t(type, id) {};
  expr_body2(const expr_body2 &ref);

public:
  virtual void convert_smt(prop_convt &obj, void *&arg) const;
  virtual expr2tc clone(void) const;
};

/** Constant class type. Not designed to contain any piece of data or method in
 *  particular, but allows for additional type safety. */
template <class derived>
class constant2t : public expr_body<derived>
{
public:
  constant2t(const type2tc type, expr2t::expr_ids id) : expr_body<derived>(type, id) {}
  constant2t(const constant2t &ref) : expr_body<derived>(ref) {};
};

/** Constant integer class. Records a constant integer of an arbitary
 *  precision */
class constant_int2t : public constant2t<constant_int2t>
{
public:
  constant_int2t(type2tc type, const BigInt &input);
  constant_int2t(const constant_int2t &ref);

  /** Accessor for fetching native int of this constant */
  unsigned long as_ulong(void) const;
  long as_long(void) const;
  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  /** Arbitary precision integer record. */
  BigInt constant_value;
};

/** Constant fixedbv class. Records a floating point number in what I assume
 *  to be mantissa/exponent form, but which is described throughout CBMC code
 *  as fraction/integer parts. */
class constant_fixedbv2t : public constant2t<constant_fixedbv2t>
{
public:
  constant_fixedbv2t(type2tc type, const fixedbvt &value);
  constant_fixedbv2t(const constant_fixedbv2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const fixedbvt value;
};

class constant_bool2t : public constant2t<constant_bool2t>
{
public:
  constant_bool2t(bool value);
  constant_bool2t(const constant_bool2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  bool is_true(void) const;
  bool is_false(void) const;

  /** Arbitary precision integer record. */
  bool constant_value;
};

/** Constant class for string constants. */
class constant_string2t : public constant2t<constant_string2t>
{
public:
  constant_string2t(const type2tc type, const std::string &stringref);
  constant_string2t(const constant_string2t &ref);

  /** Convert string to a constant length array */
  expr2tc to_array(void) const;

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  /** Arbitary precision integer record. */
  const std::string value;
};

/** Const datatype - for holding structs and unions */
template <class derived>
class constant_datatype2t : public constant2t<derived>
{
public:
  constant_datatype2t(const type2tc type, expr2t::expr_ids id,
                      const std::vector<expr2tc> &members)
    : constant2t<derived>(type, id), datatype_members(members)
  {
  }

  constant_datatype2t(const constant_datatype2t &ref)
    : constant2t<derived>(ref), datatype_members(ref.datatype_members)
  {
  }

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const std::vector<expr2tc> datatype_members;
};

class constant_struct2t : public constant_datatype2t<constant_struct2t>
{
public:
  constant_struct2t(const type2tc type, const std::vector<expr2tc> &members);
  constant_struct2t(const constant_struct2t &ref);
};

class constant_union2t : public constant_datatype2t<constant_union2t>
{
public:
  constant_union2t(const type2tc type, const std::vector<expr2tc> &members);
  constant_union2t(const constant_union2t &ref);
};

class constant_array2t : public constant2t<constant_array2t>
{
public:
  constant_array2t(const type2tc type, const std::vector<expr2tc> &members);
  constant_array2t(const constant_array2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const std::vector<expr2tc> datatype_members;
};

class constant_array_of2t : public constant2t<constant_array_of2t>
{
public:
  constant_array_of2t(const type2tc type, const expr2tc initializer);
  constant_array_of2t(const constant_array_of2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  // Type records the size of the array; this records the initializer.
  const expr2tc initializer;
};

class symbol2t : public expr_body<symbol2t>
{
public:
  symbol2t(const type2tc type, irep_idt name);
  symbol2t(const symbol2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  // Symbol name - only so long as a symbol is a string. In the future, this
  // should really really change.
  irep_idt name;
};

class typecast2t : public expr_body<typecast2t>
{
public:
  typecast2t(const type2tc type, const expr2tc expr);
  typecast2t(const typecast2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  // Expression to typecast from.
  const expr2tc from;
};

class if2t : public expr_body<if2t>
{
public:
  if2t(const type2tc type, const expr2tc cond, const expr2tc true_val,
             const expr2tc false_val);
  if2t(const if2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

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
template <class derived>
class rel2t : public expr_body<derived>
{
public:
  rel2t(expr2t::expr_ids id, const expr2tc val1, const expr2tc val2) :
    expr_body<derived>(type2tc(new bool_type2t()), id),
                       side_1(val1), side_2(val2)
  {
  }

  rel2t(const rel2t &ref) :
    expr_body<derived>(ref), side_1(ref.side_1), side_2(ref.side_2)
  {
  }

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc side_1;
  const expr2tc side_2;
};

class equality2t : public rel2t<equality2t>
{
public:
  equality2t(const expr2tc val1, const expr2tc val2);
  equality2t(const equality2t &ref);
};

class notequal2t : public rel2t<notequal2t>
{
public:
  notequal2t(const expr2tc val1, const expr2tc val2);
  notequal2t(const notequal2t &ref);
};

class lessthan2t : public rel2t<lessthan2t>
{
public:
  lessthan2t(const expr2tc val1, const expr2tc val2);
  lessthan2t(const lessthan2t &ref);
};

class greaterthan2t : public rel2t<greaterthan2t>
{
public:
  greaterthan2t(const expr2tc val1, const expr2tc val2);
  greaterthan2t(const greaterthan2t &ref);
};

class lessthanequal2t : public rel2t<lessthanequal2t>
{
public:
  lessthanequal2t(const expr2tc val1, const expr2tc val2);
  lessthanequal2t(const lessthanequal2t &ref);
};

class greaterthanequal2t : public rel2t<greaterthanequal2t>
{
public:
  greaterthanequal2t(const expr2tc val1, const expr2tc val2);
  greaterthanequal2t(const greaterthanequal2t &ref);
};

/** Logical operations base class. Base for any logical operator. No storage in
 *  this particular class. Result is always of boolean type. */
template <class derived>
class lops2t : public expr_body<derived>
{
public:
  lops2t(expr2t::expr_ids id) :
    expr_body<derived>(type2tc(new bool_type2t()), id) { }

  lops2t(const lops2t &ref) : expr_body<derived> (ref) {}
};

/** Not operator. Takes a boolean value; results in a boolean value. */
class not2t : public lops2t<not2t>
{
public:
  not2t(const expr2tc notval);
  not2t(const not2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc notvalue;
};

/** Base class for 2-operand boolean oeprators. Always results in a boolean,
 *  takes two operands, both of boolean type. */
template <class derived>
class logical_2ops2t : public lops2t<derived>
{
public:
  logical_2ops2t(expr2t::expr_ids id, const expr2tc val1, const expr2tc val2) :
    lops2t<derived>(id), side_1(val1), side_2(val2) { }
  logical_2ops2t(const logical_2ops2t &ref) : lops2t<derived>(ref),
                        side_1(ref.side_1), side_2(ref.side_2) { }

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc side_1;
  const expr2tc side_2;
};

class and2t : public logical_2ops2t<and2t>
{
public:
  and2t(const expr2tc val1, const expr2tc val2);
  and2t(const and2t &ref);
};

class or2t : public logical_2ops2t<or2t>
{
public:
  or2t(const expr2tc val1, const expr2tc val2);
  or2t(const or2t &ref);
};

class xor2t : public logical_2ops2t<xor2t>
{
public:
  xor2t(const expr2tc val1, const expr2tc val2);
  xor2t(const xor2t &ref);
};

class implies2t : public logical_2ops2t<implies2t>
{
public:
  implies2t(const expr2tc val1, const expr2tc val2);
  implies2t(const implies2t &ref);
};

/** Binary operations base class. Take a type, probably integer with a width,
 *  and some operands. */
template <class derived>
class binops2t : public expr_body<derived>
{
public:
  binops2t(const type2tc type, expr2t::expr_ids id,
           const expr2tc val1, const expr2tc val2) :
    expr_body<derived>(type, id), side_1(val1), side_2(val2) { }
  binops2t(const binops2t &ref) :
    expr_body<derived>(ref), side_1(ref.side_1), side_2(ref.side_2) { }

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc side_1;
  const expr2tc side_2;
};

class bitand2t : public binops2t<bitand2t>
{
public:
  bitand2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  bitand2t(const bitand2t &ref);
};

class bitor2t : public binops2t<bitor2t>
{
public:
  bitor2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  bitor2t(const bitor2t &ref);
};

class bitxor2t : public binops2t<bitxor2t>
{
public:
  bitxor2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  bitxor2t(const bitxor2t &ref);
};

class bitnand2t : public binops2t<bitnand2t>
{
public:
  bitnand2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  bitnand2t(const bitnand2t &ref);
};

class bitnor2t : public binops2t<bitnor2t>
{
public:
  bitnor2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  bitnor2t(const bitnor2t &ref);
};

class bitnxor2t : public binops2t<bitnxor2t>
{
public:
  bitnxor2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  bitnxor2t(const bitnxor2t &ref);
};

class lshr2t : public binops2t<lshr2t>
{
public:
  lshr2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  lshr2t(const lshr2t &ref);
};

/** Arithmatic base class. For all operations that are essentially integer
 *  arithmatic. */
template <class derived>
class arith2t : public expr_body<derived>
{
public:
  arith2t(const type2tc type, expr2t::expr_ids id)
    : expr_body<derived>(type, id) { }
  arith2t(const arith2t &ref) : expr_body<derived>(ref) { }
};

class neg2t : public arith2t<neg2t>
{
public:
  neg2t(const type2tc type, const expr2tc value);
  neg2t(const neg2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc value;
};

class abs2t : public arith2t<abs2t>
{
public:
  abs2t(const type2tc type, const expr2tc value);
  abs2t(const abs2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc value;
};

/** Base two-operand arithmatic class. */
template <class derived>
class arith_2op2t : public arith2t<derived>
{
public:
  arith_2op2t(const type2tc type, expr2t::expr_ids id,
              const expr2tc val1, const expr2tc val2)
    : arith2t<derived>(type, id), part_1(val1), part_2(val2) { }
  arith_2op2t(const arith_2op2t &ref)
    : arith2t<derived>(ref), part_1(ref.part_1), part_2(ref.part_2) { }

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc part_1;
  const expr2tc part_2;
};

class add2t : public arith_2op2t<add2t>
{
public:
  add2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  add2t(const add2t &ref);
};

class sub2t : public arith_2op2t<sub2t>
{
public:
  sub2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  sub2t(const sub2t &ref);
};

class mul2t : public arith_2op2t<mul2t>
{
public:
  mul2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  mul2t(const mul2t &ref);
};

class div2t : public arith_2op2t<div2t>
{
public:
  div2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  div2t(const div2t &ref);
};

class modulus2t : public arith_2op2t<modulus2t>
{
public:
  modulus2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  modulus2t(const modulus2t &ref);
};

class shl2t : public arith_2op2t<shl2t>
{
public:
  shl2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  shl2t(const shl2t &ref);
};

class ashr2t : public arith_2op2t<ashr2t>
{
public:
  ashr2t(const type2tc type, const expr2tc val1, const expr2tc val2);
  ashr2t(const ashr2t &ref);
};

/** Dynamic object operation. Checks to see whether or not the object is a
 *  dynamically allocated object or not. */
class dynamic_object2t : public lops2t<dynamic_object2t>
{
public:
  dynamic_object2t(const expr2tc val);
protected:
  dynamic_object2t(const dynamic_object2t &ref);

public:
  virtual expr2tc clone(void) const;

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc ptr_obj;
};

/** Same object operation. Compares two pointer objects to see if they're the
 *  same, with a boolean result. */
class same_object2t : public arith_2op2t<same_object2t>
{
public:
  same_object2t(const expr2tc val1, const expr2tc val2);
  same_object2t(const same_object2t &ref);
};

/** Pointer offset. Extract pointer offset from a pointer value. Subclass of
 *  arithmatic because it returns an integer. */
class pointer_offset2t : public arith2t<pointer_offset2t>
{
public:
  pointer_offset2t(const type2tc type, const expr2tc pointer);
  pointer_offset2t(const pointer_offset2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc pointer_obj;
};

/** Pointer object. Extract pointer object from a pointer value. Subclass of
 *  arithmatic because it returns an integer. */
class pointer_object2t : public arith2t<pointer_object2t>
{
public:
  pointer_object2t(const type2tc type, const expr2tc pointer);
  pointer_object2t(const pointer_object2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc pointer_obj;
};

class address_of2t : public arith2t<address_of2t>
{
public:
  address_of2t(const type2tc subtype, const expr2tc pointer);
  address_of2t(const address_of2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc pointer_obj;
};

/** Base class for byte operations. Endianness is a global property of the
 *  model that we're building, and we only need to care about it when we build
 *  an smt model in the end, not at any other point. */
template <class derived>
class byte_ops2t : public expr_body<derived>
{
public:
  byte_ops2t(const type2tc type, expr2t::expr_ids id)
    : expr_body<derived>(type, id) {}
  byte_ops2t(const byte_ops2t &ref)
    : expr_body<derived> (ref) {}
};

/** Data extraction from some expression. Type is whatever type we're expecting
 *  to pull out of it. source_value is whatever piece of data we're operating
 *  upon. source_offset is the _byte_ offset into source_value to extract data
 *  from. */
class byte_extract2t : public byte_ops2t<byte_extract2t>
{
public:
  byte_extract2t(const type2tc type, bool big_endian,
                 const expr2tc source, const expr2tc offs);
  byte_extract2t(const byte_extract2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  bool big_endian;
  const expr2tc source_value;
  const expr2tc source_offset;
};

/** Data insertion. Type is the type of the resulting expression. source_value
 *  is the piece of data to insert data into. source_offset is the byte offset
 *  of where to put it. udpate_value is the piece of data to shoehorn into
 *  source_value. */
class byte_update2t : public byte_ops2t<byte_update2t>
{
public:
  byte_update2t(const type2tc type, bool big_endian,
                const expr2tc source, const expr2tc offs,
                const expr2tc update_val);
  byte_update2t(const byte_update2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  bool big_endian;
  const expr2tc source_value;
  const expr2tc source_offset;
  const expr2tc update_value;
};

/** Base type of datatype operations. */
template <class derived>
class datatype_ops2t : public expr_body<derived>
{
public:
  datatype_ops2t(const type2tc type, expr2t::expr_ids id)
    : expr_body<derived>(type, id) {}
  datatype_ops2t(const datatype_ops2t &ref)
    : expr_body<derived>(ref) {}
};

/** With operation. Some kind of piece of data, another piece of data to
 *  insert into it, and where to put it. */
class with2t : public datatype_ops2t<with2t>
{
public:
  with2t(const type2tc type, const expr2tc source, const expr2tc field,
         const expr2tc update);
  with2t(const with2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc source_data;
  const expr2tc update_field;
  const expr2tc update_data;
};

/** Member operation. Extracts some field from a datatype. */
class member2t : public datatype_ops2t<member2t>
{
public:
  member2t(const type2tc type, const expr2tc source,
           const constant_string2t &member);
  member2t(const member2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc source_data;
  const constant_string2t member;
};

/** Index operation. Extracts an entry from an array. */
class index2t : public datatype_ops2t<index2t>
{
public:
  index2t(const type2tc type, const expr2tc source, const expr2tc index);
  index2t(const index2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc source_data;
  const expr2tc index;
};

/** Zero string operation. Don't quite understand it. Just operates on the
 *  string struct as far as I know. Result is boolean. */
class zero_string2t : public datatype_ops2t<zero_string2t>
{
public:
  zero_string2t(const expr2tc string);
  zero_string2t(const zero_string2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc string;
};

/** Zero length string. Unknown dirference from zero_string. */
class zero_length_string2t : public datatype_ops2t<zero_length_string2t>
{
public:
  zero_length_string2t(const expr2tc string);
  zero_length_string2t(const zero_length_string2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc string;
};

/** Isnan operation. Checks whether expression is a NaN or not. */
class isnan2t : public lops2t<isnan2t>
{
public:
  isnan2t(const expr2tc val);
  isnan2t(const isnan2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc value;
};

/** Check whether operand overflows. Operand must be either add, subtract,
 *  or multiply. XXXjmorse - in the future we should ensure the type of the
 *  operand is the expected type result of the operation. That way we can tell
 *  whether to do a signed or unsigned over/underflow test. */
class overflow2t : public lops2t<overflow2t>
{
public:
  overflow2t(const expr2tc val1);
  overflow2t(const overflow2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc operand;
};

class overflow_cast2t : public lops2t<overflow_cast2t>
{
public:
  overflow_cast2t(const expr2tc val1, unsigned int bits);
  overflow_cast2t(const overflow_cast2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc operand;
  unsigned int bits;
};

class overflow_neg2t : public lops2t<overflow_neg2t>
{
public:
  overflow_neg2t(const expr2tc val1);
  overflow_neg2t(const overflow_neg2t &ref);

  virtual bool cmp(const expr2t &ref) const;
  virtual int lt(const expr2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const;
  virtual void do_crc(boost::crc_32_type &crc) const;

  const expr2tc operand;
};

inline bool operator==(boost::shared_ptr<type2t> const & a, boost::shared_ptr<type2t> const & b)
{
  return (*a.get() == *b.get());
}

inline bool operator!=(boost::shared_ptr<type2t> const & a, boost::shared_ptr<type2t> const & b)
{
  return (*a.get() != *b.get());
}

inline bool operator<(boost::shared_ptr<type2t> const & a, boost::shared_ptr<type2t> const & b)
{
  return (*a.get() < *b.get());
}

inline bool operator>(boost::shared_ptr<type2t> const & a, boost::shared_ptr<type2t> const & b)
{
  return (*b.get() < *a.get());
}

inline bool operator==(boost::shared_ptr<expr2t> const & a, boost::shared_ptr<expr2t> const & b)
{
  return (*a.get() == *b.get());
}

inline bool operator!=(boost::shared_ptr<expr2t> const & a, boost::shared_ptr<expr2t> const & b)
{
  return (*a.get() != *b.get());
}

inline bool operator<(boost::shared_ptr<expr2t> const & a, boost::shared_ptr<expr2t> const & b)
{
  return (*a.get() < *b.get());
}

inline bool operator>(boost::shared_ptr<expr2t> const & a, boost::shared_ptr<expr2t> const & b)
{
  return (*b.get() < *a.get());
}

struct irep2_hash
{
  size_t operator()(const expr2tc &ref) const { return ref->crc(); }
};

// Same deal as for "type_macros".
#ifdef NDEBUG
#define dynamic_cast static_cast
#endif
#define expr_macros(name) \
  inline bool is_##name##2t(const expr2tc &t) \
    { return t->expr_id == expr2t::name##_id; } \
  inline bool is_##name##2t(const expr2t &r) \
    { return r.expr_id == expr2t::name##_id; } \
  inline const name##2t & to_##name##2t(const expr2tc &t) \
    { return dynamic_cast<const name##2t &> (*t.get()); } \
  inline name##2t & to_##name##2t(expr2tc &t) \
    { return dynamic_cast<name##2t &> (*t.get()); }

expr_macros(constant_int);
expr_macros(constant_fixedbv);
expr_macros(constant_bool);
expr_macros(constant_string);
expr_macros(constant_struct);
expr_macros(constant_union);
expr_macros(constant_array);
expr_macros(constant_array_of);
expr_macros(symbol);
expr_macros(typecast);
expr_macros(if);
expr_macros(equality);
expr_macros(notequal);
expr_macros(lessthan);
expr_macros(greaterthan);
expr_macros(lessthanequal);
expr_macros(greaterthanequal);
expr_macros(not);
expr_macros(and);
expr_macros(or);
expr_macros(xor);
expr_macros(bitand);
expr_macros(bitor);
expr_macros(bitxor);
expr_macros(bitnand);
expr_macros(bitnor);
expr_macros(bitnxor);
expr_macros(lshr);
expr_macros(neg);
expr_macros(abs);
expr_macros(add);
expr_macros(sub);
expr_macros(mul);
expr_macros(div);
expr_macros(modulus);
expr_macros(shl);
expr_macros(ashr);
expr_macros(dynamic_object);
expr_macros(same_object);
expr_macros(pointer_offset);
expr_macros(pointer_object);
expr_macros(address_of);
expr_macros(byte_extract);
expr_macros(byte_update);
expr_macros(with);
expr_macros(member);
expr_macros(index);
expr_macros(zero_string);
expr_macros(zero_length_string);
expr_macros(isnan);
expr_macros(overflow);
expr_macros(overflow_cast);
expr_macros(overflow_neg);
#undef expr_macros
#ifdef dynamic_cast
#undef dynamic_cast
#endif

inline bool is_constant_expr(const expr2tc &t)
{
  return t->expr_id == expr2t::constant_int_id ||
         t->expr_id == expr2t::constant_fixedbv_id ||
         t->expr_id == expr2t::constant_bool_id ||
         t->expr_id == expr2t::constant_string_id ||
         t->expr_id == expr2t::constant_struct_id ||
         t->expr_id == expr2t::constant_union_id ||
         t->expr_id == expr2t::constant_array_id ||
         t->expr_id == expr2t::constant_array_of_id;
}

// And now, some more utilities.
class type_poolt {
public:
  type_poolt(void);

  type2tc bool_type;
  type2tc empty_type;
  type2tc code_type;

  const type2tc &get_bool() const { return bool_type; }
  const type2tc &get_empty() const { return empty_type; }
  const type2tc &get_code() const { return code_type; }

  // For other types, have a pool of them for quick lookup.
  std::map<const typet, type2tc> struct_map;
  std::map<const typet, type2tc> union_map;
  std::map<const typet, type2tc> array_map;
  std::map<const typet, type2tc> pointer_map;
  std::map<const typet, type2tc> unsignedbv_map;
  std::map<const typet, type2tc> signedbv_map;
  std::map<const typet, type2tc> fixedbv_map;
  std::map<const typet, type2tc> string_map;
  std::map<const typet, type2tc> symbol_map;

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
  const type2tc &get_string(const typet &val);
  const type2tc &get_symbol(const typet &val);

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

#endif /* _UTIL_IREP2_H_ */
