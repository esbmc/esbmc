#ifndef _UTIL_IREP2_H_
#define _UTIL_IREP2_H_

#include <stdarg.h>

#include <vector>

#define BOOST_SP_DISABLE_THREADS 1

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
  for (std::vector<irep_idt>::const_iterator (it) = (vect).begin();\
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

template <class T>
static inline std::string type_to_string(const T &theval, int indent);

template <class T>
static inline bool do_type_cmp(const T &side1, const T &side2);

template <class T>
static inline int do_type_lt(const T &side1, const T &side2);

template <class T>
static inline void do_type_crc(const T &theval, boost::crc_32_type &crc);

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
  virtual void convert_smt_type(prop_convt &obj, void *&arg) const = 0;
  virtual unsigned int get_width(void) const = 0;
  bool operator==(const type2t &ref) const;
  bool operator!=(const type2t &ref) const;
  bool operator<(const type2t &ref) const;
  int ltchecked(const type2t &ref) const;
  std::string pretty(unsigned int indent = 0) const;
  void dump(void) const;
  uint32_t crc(void) const;
  bool cmpchecked(const type2t &ref) const;
  virtual bool cmp(const type2t &ref) const = 0;
  virtual int lt(const type2t &ref) const;
  virtual list_of_memberst tostring(unsigned int indent) const = 0;
  virtual void do_crc(boost::crc_32_type &crc) const;

  /** Instance of type_ids recording this types type. */
  type_ids type_id;
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
    implies_id,
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
};

// for "ESBMC templates",
namespace esbmct {

  // Template metaprogramming (vomit) -- define tag classes to instanciate
  // different class fields with different names and different types.

  #define field_name_macro(name) \
  template <class fieldtype> \
  struct name_class_##name { \
  public: \
    name_class_##name(fieldtype &someval) : name(someval) {} \
    name_class_##name(const name_class_##name &ref) : name(ref.name) {} \
    inline void tostring(list_of_memberst &membs, int indent) const \
    { return membs.push_back(member_entryt("" #name, \
                             type_to_string<fieldtype>(name, indent)));}\
    inline bool cmp(const name_class_##name &theother) const { \
      return do_type_cmp<fieldtype>(name, theother.name); }\
    inline int lt(const name_class_##name &theother) const { \
      return do_type_lt<fieldtype>(name, theother.name); }\
    inline void do_crc(boost::crc_32_type &crc) const { \
      do_type_crc<fieldtype>(name, crc); return; }\
    fieldtype name; \
  }; \
  template <class fieldtype> \
  struct name_##name { \
  public: \
    typedef name_class_##name<fieldtype> type; \
  };

  field_name_macro(constant_value);
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
  field_name_macro(initializer);
  field_name_macro(operand);
  field_name_macro(symbol_name);
  field_name_macro(members);
  field_name_macro(member_names);
  field_name_macro(width);
  field_name_macro(integer_bits);
  field_name_macro(subtype);
  field_name_macro(array_size);
  field_name_macro(size_is_infinite);
  #undef field_name_macro

  // Multiple empty name tags are required to avoid inheretance of the same type

  #define empty_names_macro(num) \
  class name_empty_##num; \
  struct name_class_empty_##num { \
  public: \
    name_class_empty_##num() { } \
    name_class_empty_##num(const name_class_empty_##num &ref \
                           __attribute__((unused))) {} \
    name_class_empty_##num(const name_empty_##num &ref \
                           __attribute__((unused))) {} \
    inline void tostring(list_of_memberst &membs __attribute__((unused)),\
                         int indent __attribute__((unused))) const\
    { return; } \
    inline bool cmp(const name_class_empty_##num &ref __attribute__((unused)))\
    const { return true; }\
    inline int lt(const name_class_empty_##num &ref __attribute__((unused)))\
    const { return 0; }\
    inline void do_crc(boost::crc_32_type &crc __attribute__((unused))) const \
    { return; }\
  }; \
  class name_empty_##num { \
  public: \
    typedef name_class_empty_##num type; \
  };

  empty_names_macro(1);
  empty_names_macro(2);
  empty_names_macro(3);
  empty_names_macro(4);
  #undef empty_names_macro

  // Type tags
  #define field_type_macro(name, thetype) \
  class name { \
    public: \
    typedef thetype type; \
  };

  field_type_macro(int_type_tag, int);
  field_type_macro(uint_type_tag, unsigned int);
  field_type_macro(bool_type_tag, bool);
  field_type_macro(bigint_type_tag, BigInt);
  field_type_macro(expr2tc_type_tag, expr2tc);
  field_type_macro(fixedbv_type_tag, fixedbvt);
  field_type_macro(expr2tc_vec_type_tag, std::vector<expr2tc>);
  field_type_macro(irepidt_type_tag, irep_idt);
  field_type_macro(type2tc_vec_type_tag, std::vector<type2tc>);
  field_type_macro(irepidt_vec_type_tag, std::vector<irep_idt>);
  field_type_macro(type2tc_type_tag, type2tc);
  #undef field_type_macro

  #define member_record_macro(thename, thetype, fieldname) \
  struct thename { \
    typedef fieldname<thetype::type>::type fieldtype; \
    typedef thetype::type type; \
    const static int enabled = true; \
  };

  member_record_macro(constant_int_value, int_type_tag, name_constant_value);
  member_record_macro(constant_bigint_value, bigint_type_tag,
                      name_constant_value);
  member_record_macro(fixedbv_value, fixedbv_type_tag, name_value);
  member_record_macro(constant_bool_value, bool_type_tag, name_constant_value);
  member_record_macro(irepidt_value, irepidt_type_tag, name_value);
  member_record_macro(expr2tc_vec_datatype_members, expr2tc_vec_type_tag,
                      name_datatype_members);
  member_record_macro(expr2tc_initializer, expr2tc_type_tag, name_initializer);
  member_record_macro(irepidt_name, irepidt_type_tag, name_name);
  member_record_macro(expr2tc_from, expr2tc_type_tag, name_from);
  member_record_macro(expr2tc_cond, expr2tc_type_tag, name_cond);
  member_record_macro(expr2tc_true_value, expr2tc_type_tag, name_true_value);
  member_record_macro(expr2tc_false_value, expr2tc_type_tag, name_false_value);
  member_record_macro(expr2tc_side_1, expr2tc_type_tag, name_side_1);
  member_record_macro(expr2tc_side_2, expr2tc_type_tag, name_side_2);
  member_record_macro(expr2tc_value, expr2tc_type_tag, name_value);
  member_record_macro(expr2tc_ptr_obj, expr2tc_type_tag, name_ptr_obj);
  member_record_macro(bool_big_endian, bool_type_tag, name_big_endian);
  member_record_macro(expr2tc_source_value, expr2tc_type_tag,
                      name_source_value);
  member_record_macro(expr2tc_source_offset, expr2tc_type_tag,
                      name_source_offset);
  member_record_macro(expr2tc_update_value, expr2tc_type_tag,
                      name_update_value);
  member_record_macro(expr2tc_update_field, expr2tc_type_tag,
                      name_update_field);
  member_record_macro(irepidt_member, irepidt_type_tag, name_member);
  member_record_macro(expr2tc_index, expr2tc_type_tag, name_index);
  member_record_macro(expr2tc_string, expr2tc_type_tag, name_string);
  member_record_macro(expr2tc_operand, expr2tc_type_tag, name_operand);
  member_record_macro(uint_bits, uint_type_tag, name_bits);
  member_record_macro(irepidt_symbol_name, irepidt_type_tag, name_symbol_name);
  member_record_macro(type2tc_vec_members, type2tc_vec_type_tag, name_members);
  member_record_macro(irepidt_vec_member_names, irepidt_vec_type_tag,
                      name_member_names);
  member_record_macro(uint_width, uint_type_tag, name_width);
  member_record_macro(uint_int_bits, uint_type_tag, name_integer_bits);
  member_record_macro(type2tc_subtype, type2tc_type_tag, name_subtype);
  member_record_macro(expr2tc_array_size, expr2tc_type_tag, name_array_size);
  member_record_macro(bool_size_is_inf, bool_type_tag, name_size_is_infinite);
  #undef member_record_macro

  template <class thename>
  struct blank_value {
    typedef typename thename::type fieldtype;
    typedef thename type;
    const static thename defaultval;
    const static int enabled = false;
  };

  template <class derived,
            class field1 = esbmct::blank_value<esbmct::name_empty_1>,
            class field2 = esbmct::blank_value<esbmct::name_empty_2>,
            class field3 = esbmct::blank_value<esbmct::name_empty_3>,
            class field4 = esbmct::blank_value<esbmct::name_empty_4> >
  class expr :
    public expr2t,
    public field1::fieldtype,
    public field2::fieldtype,
    public field3::fieldtype,
    public field4::fieldtype
  {
  public:

    expr(const type2tc type, expr_ids id,
        typename field1::type arg1 = field1::defaultval,
        typename field2::type arg2 = field2::defaultval,
        typename field3::type arg3 = field3::defaultval,
        typename field4::type arg4 = field4::defaultval)
      : expr2t(type, id),
        field1::fieldtype(arg1),
        field2::fieldtype(arg2),
        field3::fieldtype(arg3),
        field4::fieldtype(arg4)
    {};

    expr(const expr &ref)
      : expr2t(ref),
        field1::fieldtype(ref),
        field2::fieldtype(ref),
        field3::fieldtype(ref),
        field4::fieldtype(ref)
    {}

    virtual void convert_smt(prop_convt &obj, void *&arg) const;
    virtual expr2tc clone(void) const;
    virtual list_of_memberst tostring(unsigned int indent) const;
    virtual bool cmp(const expr2t &ref) const;
    virtual int lt(const expr2t &ref) const;
    virtual void do_crc(boost::crc_32_type &crc) const;
  };

  template <class derived,
            class field1 = esbmct::blank_value<esbmct::name_empty_1>,
            class field2 = esbmct::blank_value<esbmct::name_empty_2>,
            class field3 = esbmct::blank_value<esbmct::name_empty_3>,
            class field4 = esbmct::blank_value<esbmct::name_empty_4> >
  class type :
    public type2t,
    public field1::fieldtype,
    public field2::fieldtype,
    public field3::fieldtype,
    public field4::fieldtype
  {
  public:

    type(type_ids id,
        typename field1::type arg1 = field1::defaultval,
        typename field2::type arg2 = field2::defaultval,
        typename field3::type arg3 = field3::defaultval,
        typename field4::type arg4 = field4::defaultval)
      : type2t(id),
        field1::fieldtype(arg1),
        field2::fieldtype(arg2),
        field3::fieldtype(arg3),
        field4::fieldtype(arg4)
    {};

    type(const type &ref)
      : type2t(ref),
        field1::fieldtype(ref),
        field2::fieldtype(ref),
        field3::fieldtype(ref),
        field4::fieldtype(ref)
    {}

    virtual void convert_smt_type(prop_convt &obj, void *&arg) const;
    virtual type2tc clone(void) const;
    virtual list_of_memberst tostring(unsigned int indent) const;
    virtual bool cmp(const type2t &ref) const;
    virtual int lt(const type2t &ref) const;
    virtual void do_crc(boost::crc_32_type &crc) const;
  };
}; // esbmct

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
class bool_type2t : public esbmct::type<bool_type2t>
{
public:
  bool_type2t(void) : esbmct::type<bool_type2t>(bool_id) {}
  bool_type2t(const bool_type2t &ref) : esbmct::type<bool_type2t>(ref) {}
  virtual unsigned int get_width(void) const;
};
template class esbmct::type<bool_type2t>;

/** Empty type. For void pointers and the like, with no type. No extra data */
class empty_type2t : public esbmct::type<empty_type2t>
{
public:
  empty_type2t(void) : esbmct::type<empty_type2t>(empty_id) {}
  empty_type2t(const empty_type2t &ref) : esbmct::type<empty_type2t>(ref) {}
  virtual unsigned int get_width(void) const;
};
template class esbmct::type<empty_type2t>;

/** Symbol type. Temporary, prior to linking up types after parsing, or when
 *  a struct/array contains a recursive pointer to its own type. */
class symbol_type2t : public esbmct::type<symbol_type2t,
                                           esbmct::irepidt_symbol_name>
{
public:
  symbol_type2t(const dstring sym_name)
    : esbmct::type<symbol_type2t, esbmct::irepidt_symbol_name>
      (symbol_id, sym_name) { }
  symbol_type2t(const symbol_type2t &ref)
    : esbmct::type<symbol_type2t, esbmct::irepidt_symbol_name>
      (ref) { }
  virtual unsigned int get_width(void) const;
};
template class esbmct::type<symbol_type2t, esbmct::irepidt_symbol_name>;

class struct_type2t : public esbmct::type<struct_type2t,
                                          esbmct::type2tc_vec_members,
                                          esbmct::irepidt_vec_member_names,
                                          esbmct::irepidt_name>
{
public:
  struct_type2t(std::vector<type2tc> &members, std::vector<irep_idt> memb_names,
                irep_idt name)
    : esbmct::type<struct_type2t, esbmct::type2tc_vec_members,
                   esbmct::irepidt_vec_member_names, esbmct::irepidt_name>
      (struct_id, members, memb_names, name) {}

  struct_type2t(const struct_type2t &ref)
    : esbmct::type<struct_type2t, esbmct::type2tc_vec_members,
                   esbmct::irepidt_vec_member_names, esbmct::irepidt_name>
      (ref) {}

  virtual unsigned int get_width(void) const;
};
template class esbmct::type<struct_type2t, esbmct::type2tc_vec_members,
                        esbmct::irepidt_vec_member_names, esbmct::irepidt_name>;

class union_type2t : public esbmct::type<union_type2t,
                                          esbmct::type2tc_vec_members,
                                          esbmct::irepidt_vec_member_names,
                                          esbmct::irepidt_name>
{
public:
  union_type2t(std::vector<type2tc> &members, std::vector<irep_idt> memb_names,
                irep_idt name)
    : esbmct::type<union_type2t, esbmct::type2tc_vec_members,
                   esbmct::irepidt_vec_member_names, esbmct::irepidt_name>
      (union_id, members, memb_names, name) {}

  union_type2t(const union_type2t &ref)
    : esbmct::type<union_type2t, esbmct::type2tc_vec_members,
                   esbmct::irepidt_vec_member_names, esbmct::irepidt_name>
      (ref) {}

  virtual unsigned int get_width(void) const;
};
template class esbmct::type<union_type2t, esbmct::type2tc_vec_members,
                        esbmct::irepidt_vec_member_names, esbmct::irepidt_name>;

class unsignedbv_type2t : public esbmct::type<unsignedbv_type2t,
                                          esbmct::uint_width>
{
public:
  unsignedbv_type2t(unsigned int width)
    : esbmct::type<unsignedbv_type2t, esbmct::uint_width>
      (unsignedbv_id, width) { }
  unsignedbv_type2t(const unsignedbv_type2t &ref)
    : esbmct::type<unsignedbv_type2t, esbmct::uint_width>(ref) { }
  virtual unsigned int get_width(void) const;
};
template class esbmct::type<unsignedbv_type2t, esbmct::uint_width>;

class signedbv_type2t : public esbmct::type<signedbv_type2t,
                                            esbmct::uint_width>
{
public:
  signedbv_type2t(signed int width)
    : esbmct::type<signedbv_type2t, esbmct::uint_width>
      (signedbv_id, width) { }
  signedbv_type2t(const signedbv_type2t &ref)
    : esbmct::type<signedbv_type2t, esbmct::uint_width>(ref) { }
  virtual unsigned int get_width(void) const;
};
template class esbmct::type<signedbv_type2t, esbmct::uint_width>;

/** Empty type. For void pointers and the like, with no type. No extra data */
class code_type2t : public esbmct::type<code_type2t>
{
public:
  code_type2t(void) : esbmct::type<code_type2t>(code_id) {}
  code_type2t(const code_type2t &ref) : esbmct::type<code_type2t>(ref) {}
  virtual unsigned int get_width(void) const;
};
template class esbmct::type<code_type2t>;

/** Array type. Comes with a subtype of the array and a size that might be
 *  constant, might be nondeterministic. */
class array_type2t : public esbmct::type<array_type2t,
                                         esbmct::type2tc_subtype,
                                         esbmct::expr2tc_array_size,
                                         esbmct::bool_size_is_inf>
{
public:
  array_type2t(const type2tc subtype, const expr2tc size, bool inf)
    : esbmct::type<array_type2t, esbmct::type2tc_subtype,
                   esbmct::expr2tc_array_size, esbmct::bool_size_is_inf>
      (array_id, subtype, size, inf) { }
  array_type2t(const array_type2t &ref)
    : esbmct::type<array_type2t, esbmct::type2tc_subtype,
                   esbmct::expr2tc_array_size, esbmct::bool_size_is_inf>
      (ref) { }
  virtual unsigned int get_width(void) const;

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
};
template class esbmct::type<array_type2t, esbmct::type2tc_subtype,
                          esbmct::expr2tc_array_size, esbmct::bool_size_is_inf>;

/** Pointer type. Simply has a subtype, of what it points to. No other
 *  attributes */
class pointer_type2t : public esbmct::type<pointer_type2t,
                                           esbmct::type2tc_subtype>
{
public:
  pointer_type2t(const type2tc subtype)
    : esbmct::type<pointer_type2t, esbmct::type2tc_subtype>
      (pointer_id, subtype) {}
  pointer_type2t(const pointer_type2t &ref)
    : esbmct::type<pointer_type2t, esbmct::type2tc_subtype>(ref) {}
  virtual unsigned int get_width(void) const;
};
template class esbmct::type<pointer_type2t, esbmct::type2tc_subtype>;

class fixedbv_type2t : public esbmct::type<fixedbv_type2t, esbmct::uint_width,
                                           esbmct::uint_int_bits>
{
public:
  fixedbv_type2t(unsigned int width, unsigned int integer)
    : esbmct::type<fixedbv_type2t, esbmct::uint_width, esbmct::uint_int_bits>
      (fixedbv_id, width, integer) { }
  fixedbv_type2t(const fixedbv_type2t &ref)
    : esbmct::type<fixedbv_type2t, esbmct::uint_width, esbmct::uint_int_bits>
      (ref) { }
  virtual unsigned int get_width(void) const;
};
template class esbmct::type<fixedbv_type2t, esbmct::uint_width,
                             esbmct::uint_int_bits>;

class string_type2t : public esbmct::type<string_type2t, esbmct::uint_width>
{
public:
  string_type2t(unsigned int elements)
    : esbmct::type<string_type2t, esbmct::uint_width>(string_id, elements) { }
  string_type2t(const string_type2t &ref)
    : esbmct::type<string_type2t, esbmct::uint_width>(ref) { }
  virtual unsigned int get_width(void) const;
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

/** Constant integer class. Records a constant integer of an arbitary
 *  precision */
class constant_int2t : public esbmct::expr<constant_int2t,
                                         esbmct::constant_bigint_value>
{
public:
  constant_int2t(const type2tc &type, const BigInt &input)
    : esbmct::expr<constant_int2t, esbmct::constant_bigint_value>
               (type, constant_int_id, input) { }
  constant_int2t(const constant_int2t &ref)
    : esbmct::expr<constant_int2t, esbmct::constant_bigint_value> (ref) { }

  /** Accessor for fetching native int of this constant */
  unsigned long as_ulong(void) const;
  long as_long(void) const;
};
template class esbmct::expr<constant_int2t, esbmct::constant_bigint_value>;

/** Constant fixedbv class. Records a floating point number in what I assume
 *  to be mantissa/exponent form, but which is described throughout CBMC code
 *  as fraction/integer parts. */
class constant_fixedbv2t : public esbmct::expr<constant_fixedbv2t,
                                             esbmct::fixedbv_value>
{
public:
  constant_fixedbv2t(const type2tc &type, const fixedbvt &value)
    : esbmct::expr<constant_fixedbv2t, esbmct::fixedbv_value>
                (type, constant_fixedbv_id, value) { }
  constant_fixedbv2t(const constant_fixedbv2t &ref)
    : esbmct::expr<constant_fixedbv2t, esbmct::fixedbv_value> (ref) { }
};
template class esbmct::expr<constant_fixedbv2t, esbmct::fixedbv_value>;

class constant_bool2t : public esbmct::expr<constant_bool2t,
                                          esbmct::constant_bool_value>
{
public:
  constant_bool2t(bool value)
    : esbmct::expr<constant_bool2t, esbmct::constant_bool_value>
                (type_pool.get_bool(), constant_bool_id, value) { }
  constant_bool2t(const constant_bool2t &ref)
    : esbmct::expr<constant_bool2t, esbmct::constant_bool_value>(ref) { }

  bool is_true(void) const;
  bool is_false(void) const;
};
template class esbmct::expr<constant_bool2t, esbmct::constant_bool_value>;

/** Constant class for string constants. */
class constant_string2t : public esbmct::expr<constant_string2t,
                                            esbmct::irepidt_value>
{
public:
  constant_string2t(const type2tc &type, const irep_idt &stringref)
    : esbmct::expr<constant_string2t, esbmct::irepidt_value>
      (type, constant_string_id, stringref) { }
  constant_string2t(const constant_string2t &ref)
    : esbmct::expr<constant_string2t, esbmct::irepidt_value>(ref) { }

  /** Convert string to a constant length array */
  expr2tc to_array(void) const;
};
template class esbmct::expr<constant_string2t, esbmct::irepidt_value>;

class constant_struct2t : public esbmct::expr<constant_struct2t,
                                           esbmct::expr2tc_vec_datatype_members>
{
public:
  constant_struct2t(const type2tc &type, const std::vector<expr2tc> &members)
    : esbmct::expr<constant_struct2t, esbmct::expr2tc_vec_datatype_members>
      (type, constant_struct_id, members) { }
  constant_struct2t(const constant_struct2t &ref)
    : esbmct::expr<constant_struct2t, esbmct::expr2tc_vec_datatype_members>(ref){}
};
template class esbmct::expr<constant_struct2t, esbmct::expr2tc_vec_datatype_members>;

class constant_union2t : public esbmct::expr<constant_union2t,
                                           esbmct::expr2tc_vec_datatype_members>
{
public:
  constant_union2t(const type2tc &type, const std::vector<expr2tc> &members)
    : esbmct::expr<constant_union2t, esbmct::expr2tc_vec_datatype_members>
      (type, constant_union_id, members) { }
  constant_union2t(const constant_union2t &ref)
    : esbmct::expr<constant_union2t, esbmct::expr2tc_vec_datatype_members>(ref){}
};
template class esbmct::expr<constant_union2t, esbmct::expr2tc_vec_datatype_members>;

class constant_array2t : public esbmct::expr<constant_array2t,
                                           esbmct::expr2tc_vec_datatype_members>
{
public:
  constant_array2t(const type2tc &type, const std::vector<expr2tc> &members)
    : esbmct::expr<constant_array2t, esbmct::expr2tc_vec_datatype_members>
      (type, constant_array_id, members) { }
  constant_array2t(const constant_array2t &ref)
    : esbmct::expr<constant_array2t, esbmct::expr2tc_vec_datatype_members>(ref){}
};
template class esbmct::expr<constant_array2t, esbmct::expr2tc_vec_datatype_members>;

class constant_array_of2t : public esbmct::expr<constant_array_of2t,
                                              esbmct::expr2tc_initializer>
{
public:
  constant_array_of2t(const type2tc &type, const expr2tc &init)
    : esbmct::expr<constant_array_of2t, esbmct::expr2tc_initializer>
      (type, constant_array_of_id, init) { }
  constant_array_of2t(const constant_array_of2t &ref)
    : esbmct::expr<constant_array_of2t, esbmct::expr2tc_initializer>(ref){}
};
template class esbmct::expr<constant_array_of2t, esbmct::expr2tc_initializer>;

class symbol2t : public esbmct::expr<symbol2t, esbmct::irepidt_name>
{
public:
  symbol2t(const type2tc &type, const irep_idt &init)
    : esbmct::expr<symbol2t, esbmct::irepidt_name> (type, symbol_id, init) { }
  symbol2t(const symbol2t &ref)
    : esbmct::expr<symbol2t, esbmct::irepidt_name>(ref){}
};
template class esbmct::expr<symbol2t, esbmct::irepidt_name>;

class typecast2t : public esbmct::expr<typecast2t, esbmct::expr2tc_from>
{
public:
  typecast2t(const type2tc &type, const expr2tc &from)
    : esbmct::expr<typecast2t, esbmct::expr2tc_from> (type, typecast_id, from) { }
  typecast2t(const typecast2t &ref)
    : esbmct::expr<typecast2t, esbmct::expr2tc_from>(ref){}
};
template class esbmct::expr<typecast2t, esbmct::expr2tc_from>;

class if2t : public esbmct::expr<if2t,
                               esbmct::expr2tc_cond,
                               esbmct::expr2tc_true_value,
                               esbmct::expr2tc_false_value>
{
public:
  if2t(const type2tc &type, const expr2tc &cond, const expr2tc &trueval,
       const expr2tc &falseval)
    : esbmct::expr<if2t, esbmct::expr2tc_cond, esbmct::expr2tc_true_value,
                 esbmct::expr2tc_false_value>
                 (type, if_id, cond, trueval, falseval) {}
  if2t(const if2t &ref)
    : esbmct::expr<if2t, esbmct::expr2tc_cond, esbmct::expr2tc_true_value,
                 esbmct::expr2tc_false_value>(ref) {}
};
template class esbmct::expr<if2t, esbmct::expr2tc_cond,
                          esbmct::expr2tc_true_value,
                          esbmct::expr2tc_false_value>;

class equality2t : public esbmct::expr<equality2t, esbmct::expr2tc_side_1,
                                     esbmct::expr2tc_side_2>
{
public:
  equality2t(const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<equality2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type_pool.get_bool(), equality_id, v1, v2) {}
  equality2t(const equality2t &ref)
    : esbmct::expr<equality2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<equality2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>;

class notequal2t : public esbmct::expr<notequal2t, esbmct::expr2tc_side_1,
                                     esbmct::expr2tc_side_2>
{
public:
  notequal2t(const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<notequal2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type_pool.get_bool(), notequal_id, v1, v2) {}
  notequal2t(const notequal2t &ref)
    : esbmct::expr<notequal2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<notequal2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>;

class lessthan2t : public esbmct::expr<lessthan2t, esbmct::expr2tc_side_1,
                                     esbmct::expr2tc_side_2>
{
public:
  lessthan2t(const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<lessthan2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type_pool.get_bool(), lessthan_id, v1, v2) {}
  lessthan2t(const lessthan2t &ref)
    : esbmct::expr<lessthan2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<lessthan2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>;

class greaterthan2t : public esbmct::expr<greaterthan2t, esbmct::expr2tc_side_1,
                                     esbmct::expr2tc_side_2>
{
public:
  greaterthan2t(const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<greaterthan2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type_pool.get_bool(), greaterthan_id, v1, v2) {}
  greaterthan2t(const greaterthan2t &ref)
    : esbmct::expr<greaterthan2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<greaterthan2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>;

class lessthanequal2t : public esbmct::expr<lessthanequal2t,
                                          esbmct::expr2tc_side_1,
                                          esbmct::expr2tc_side_2>
{
public:
  lessthanequal2t(const expr2tc &v1, const expr2tc &v2)
   : esbmct::expr<lessthanequal2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type_pool.get_bool(), lessthanequal_id, v1, v2) {}
  lessthanequal2t(const lessthanequal2t &ref)
   : esbmct::expr<lessthanequal2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<lessthanequal2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>;

class greaterthanequal2t : public esbmct::expr<greaterthanequal2t,
                                          esbmct::expr2tc_side_1,
                                          esbmct::expr2tc_side_2>
{
public:
  greaterthanequal2t(const expr2tc &v1, const expr2tc &v2)
  : esbmct::expr<greaterthanequal2t,esbmct::expr2tc_side_1,esbmct::expr2tc_side_2>
      (type_pool.get_bool(), greaterthanequal_id, v1, v2) {}
  greaterthanequal2t(const greaterthanequal2t &ref)
  : esbmct::expr<greaterthanequal2t,esbmct::expr2tc_side_1,esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<greaterthanequal2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>;

class not2t : public esbmct::expr<not2t, esbmct::expr2tc_value>
{
public:
  not2t(const expr2tc &val)
  : esbmct::expr<not2t, esbmct::expr2tc_value>
    (type_pool.get_bool(), not_id, val) {}
  not2t(const not2t &ref)
  : esbmct::expr<not2t, esbmct::expr2tc_value>
    (ref) {}
};
template class esbmct::expr<not2t, esbmct::expr2tc_value>;

class and2t : public esbmct::expr<and2t, esbmct::expr2tc_side_1,
                                       esbmct::expr2tc_side_2>
{
public:
  and2t(const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<and2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type_pool.get_bool(), and_id, v1, v2) {}
  and2t(const and2t &ref)
    : esbmct::expr<and2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<and2t, esbmct::expr2tc_side_1,esbmct::expr2tc_side_2>;

class or2t : public esbmct::expr<or2t, esbmct::expr2tc_side_1,
                                     esbmct::expr2tc_side_2>
{
public:
  or2t(const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<or2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type_pool.get_bool(), or_id, v1, v2) {}
  or2t(const or2t &ref)
    : esbmct::expr<or2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<or2t, esbmct::expr2tc_side_1,esbmct::expr2tc_side_2>;

class xor2t : public esbmct::expr<xor2t, esbmct::expr2tc_side_1,
                                       esbmct::expr2tc_side_2>
{
public:
  xor2t(const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<xor2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type_pool.get_bool(), xor_id, v1, v2) {}
  xor2t(const xor2t &ref)
    : esbmct::expr<xor2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<xor2t, esbmct::expr2tc_side_1,esbmct::expr2tc_side_2>;

class implies2t : public esbmct::expr<implies2t, esbmct::expr2tc_side_1,
                                               esbmct::expr2tc_side_2>
{
public:
  implies2t(const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<implies2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type_pool.get_bool(), implies_id, v1, v2) {}
  implies2t(const implies2t &ref)
    : esbmct::expr<implies2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<implies2t, esbmct::expr2tc_side_1,
                                     esbmct::expr2tc_side_2>;

class bitand2t : public esbmct::expr<bitand2t, esbmct::expr2tc_side_1,
                                             esbmct::expr2tc_side_2>
{
public:
  bitand2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<bitand2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, bitand_id, v1, v2) {}
  bitand2t(const bitand2t &ref)
    : esbmct::expr<bitand2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<bitand2t, esbmct::expr2tc_side_1,
                                    esbmct::expr2tc_side_2>;

class bitor2t : public esbmct::expr<bitor2t, esbmct::expr2tc_side_1,
                                           esbmct::expr2tc_side_2>
{
public:
  bitor2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<bitor2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, bitor_id, v1, v2) {}
  bitor2t(const bitor2t &ref)
    : esbmct::expr<bitor2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<bitor2t, esbmct::expr2tc_side_1,
                                   esbmct::expr2tc_side_2>;

class bitxor2t : public esbmct::expr<bitxor2t, esbmct::expr2tc_side_1,
                                             esbmct::expr2tc_side_2>
{
public:
  bitxor2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<bitxor2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, bitxor_id, v1, v2) {}
  bitxor2t(const bitxor2t &ref)
    : esbmct::expr<bitxor2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<bitxor2t, esbmct::expr2tc_side_1,
                                    esbmct::expr2tc_side_2>;

class bitnand2t : public esbmct::expr<bitnand2t, esbmct::expr2tc_side_1,
                                               esbmct::expr2tc_side_2>
{
public:
  bitnand2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<bitnand2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, bitnand_id, v1, v2) {}
  bitnand2t(const bitnand2t &ref)
    : esbmct::expr<bitnand2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<bitnand2t, esbmct::expr2tc_side_1,
                                     esbmct::expr2tc_side_2>;

class bitnor2t : public esbmct::expr<bitnor2t, esbmct::expr2tc_side_1,
                                             esbmct::expr2tc_side_2>
{
public:
  bitnor2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<bitnor2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, bitnor_id, v1, v2) {}
  bitnor2t(const bitnor2t &ref)
    : esbmct::expr<bitnor2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<bitnor2t, esbmct::expr2tc_side_1,
                                    esbmct::expr2tc_side_2>;

class bitnxor2t : public esbmct::expr<bitnxor2t, esbmct::expr2tc_side_1,
                                               esbmct::expr2tc_side_2>
{
public:
  bitnxor2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<bitnxor2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, bitnxor_id, v1, v2) {}
  bitnxor2t(const bitnxor2t &ref)
    : esbmct::expr<bitnxor2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<bitnxor2t, esbmct::expr2tc_side_1,
                                     esbmct::expr2tc_side_2>;

class lshr2t : public esbmct::expr<lshr2t, esbmct::expr2tc_side_1,
                                         esbmct::expr2tc_side_2>
{
public:
  lshr2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<lshr2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, lshr_id, v1, v2) {}
  lshr2t(const lshr2t &ref)
    : esbmct::expr<lshr2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<lshr2t, esbmct::expr2tc_side_1,
                                  esbmct::expr2tc_side_2>;

class neg2t : public esbmct::expr<neg2t,esbmct::expr2tc_value>
{
public:
  neg2t(const type2tc &type, const expr2tc &val)
    : esbmct::expr<neg2t, esbmct::expr2tc_value> (type, neg_id, val) {}
  neg2t(const neg2t &ref)
    : esbmct::expr<neg2t, esbmct::expr2tc_value> (ref) {}
};
template class esbmct::expr<neg2t, esbmct::expr2tc_value>;

class abs2t : public esbmct::expr<abs2t,esbmct::expr2tc_value>
{
public:
  abs2t(const type2tc &type, const expr2tc &val)
    : esbmct::expr<abs2t, esbmct::expr2tc_value> (type, abs_id, val) {}
  abs2t(const abs2t &ref)
    : esbmct::expr<abs2t, esbmct::expr2tc_value> (ref) {}
};
template class esbmct::expr<abs2t, esbmct::expr2tc_value>;

class add2t : public esbmct::expr<add2t, esbmct::expr2tc_side_1,
                                       esbmct::expr2tc_side_2>
{
public:
  add2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<add2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, add_id, v1, v2) {}
  add2t(const add2t &ref)
    : esbmct::expr<add2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<add2t, esbmct::expr2tc_side_1,
                                 esbmct::expr2tc_side_2>;

class sub2t : public esbmct::expr<sub2t, esbmct::expr2tc_side_1,
                                       esbmct::expr2tc_side_2>
{
public:
  sub2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<sub2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, sub_id, v1, v2) {}
  sub2t(const sub2t &ref)
    : esbmct::expr<sub2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<sub2t, esbmct::expr2tc_side_1,
                                 esbmct::expr2tc_side_2>;

class mul2t : public esbmct::expr<mul2t, esbmct::expr2tc_side_1,
                                       esbmct::expr2tc_side_2>
{
public:
  mul2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<mul2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, mul_id, v1, v2) {}
  mul2t(const mul2t &ref)
    : esbmct::expr<mul2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<mul2t, esbmct::expr2tc_side_1,
                                 esbmct::expr2tc_side_2>;

class div2t : public esbmct::expr<div2t, esbmct::expr2tc_side_1,
                                       esbmct::expr2tc_side_2>
{
public:
  div2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<div2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, div_id, v1, v2) {}
  div2t(const div2t &ref)
    : esbmct::expr<div2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<div2t, esbmct::expr2tc_side_1,
                                 esbmct::expr2tc_side_2>;

class modulus2t : public esbmct::expr<modulus2t, esbmct::expr2tc_side_1,
                                               esbmct::expr2tc_side_2>
{
public:
  modulus2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<modulus2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, modulus_id, v1, v2) {}
  modulus2t(const modulus2t &ref)
    : esbmct::expr<modulus2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<modulus2t, esbmct::expr2tc_side_1,
                                     esbmct::expr2tc_side_2>;

class shl2t : public esbmct::expr<shl2t, esbmct::expr2tc_side_1,
                                       esbmct::expr2tc_side_2>
{
public:
  shl2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<shl2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, shl_id, v1, v2) {}
  shl2t(const shl2t &ref)
    : esbmct::expr<shl2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<shl2t, esbmct::expr2tc_side_1,
                                 esbmct::expr2tc_side_2>;

class ashr2t : public esbmct::expr<ashr2t, esbmct::expr2tc_side_1,
                                         esbmct::expr2tc_side_2>
{
public:
  ashr2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<ashr2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type, ashr_id, v1, v2) {}
  ashr2t(const ashr2t &ref)
    : esbmct::expr<ashr2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<ashr2t, esbmct::expr2tc_side_1,
                                  esbmct::expr2tc_side_2>;

class same_object2t : public esbmct::expr<same_object2t, esbmct::expr2tc_side_1,
                                                       esbmct::expr2tc_side_2>
{
public:
  same_object2t(const expr2tc &v1, const expr2tc &v2)
    : esbmct::expr<same_object2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (type_pool.get_bool(), same_object_id, v1, v2) {}
  same_object2t(const same_object2t &ref)
    : esbmct::expr<same_object2t, esbmct::expr2tc_side_1, esbmct::expr2tc_side_2>
      (ref) {}
};
template class esbmct::expr<same_object2t, esbmct::expr2tc_side_1,
                                         esbmct::expr2tc_side_2>;


class pointer_offset2t : public esbmct::expr<pointer_offset2t,
                                           esbmct::expr2tc_ptr_obj>
{
public:
  pointer_offset2t(const type2tc &type, const expr2tc &ptrobj)
    : esbmct::expr<pointer_offset2t, esbmct::expr2tc_ptr_obj>
      (type, pointer_offset_id, ptrobj) {}
  pointer_offset2t(const pointer_offset2t &ref)
    : esbmct::expr<pointer_offset2t, esbmct::expr2tc_ptr_obj> (ref) {}
};
template class esbmct::expr<pointer_offset2t, esbmct::expr2tc_ptr_obj>;

class pointer_object2t : public esbmct::expr<pointer_object2t,
                                           esbmct::expr2tc_ptr_obj>
{
public:
  pointer_object2t(const type2tc &type, const expr2tc &ptrobj)
    : esbmct::expr<pointer_object2t, esbmct::expr2tc_ptr_obj>
      (type, pointer_object_id, ptrobj) {}
  pointer_object2t(const pointer_object2t &ref)
    : esbmct::expr<pointer_object2t, esbmct::expr2tc_ptr_obj> (ref) {}
};
template class esbmct::expr<pointer_object2t, esbmct::expr2tc_ptr_obj>;

class address_of2t : public esbmct::expr<address_of2t,
                                           esbmct::expr2tc_ptr_obj>
{
public:
  address_of2t(const type2tc &subtype, const expr2tc &ptrobj)
    : esbmct::expr<address_of2t, esbmct::expr2tc_ptr_obj>
      (type2tc(new pointer_type2t(subtype)), address_of_id, ptrobj) {}
  address_of2t(const address_of2t &ref)
    : esbmct::expr<address_of2t, esbmct::expr2tc_ptr_obj> (ref) {}
};
template class esbmct::expr<address_of2t, esbmct::expr2tc_ptr_obj>;

class byte_extract2t : public esbmct::expr<byte_extract2t,
                                           esbmct::bool_big_endian,
                                           esbmct::expr2tc_source_value,
                                           esbmct::expr2tc_source_offset>
{
public:
  byte_extract2t(const type2tc &type, bool is_big_endian, const expr2tc &source,
                 const expr2tc &offset)
    : esbmct::expr<byte_extract2t, esbmct::bool_big_endian,
                 esbmct::expr2tc_source_value, esbmct::expr2tc_source_offset>
      (type, byte_extract_id, is_big_endian, source, offset) {}
  byte_extract2t(const byte_extract2t &ref)
    : esbmct::expr<byte_extract2t, esbmct::bool_big_endian,
                 esbmct::expr2tc_source_value, esbmct::expr2tc_source_offset>
      (ref) {}
};
template class esbmct::expr<byte_extract2t, esbmct::bool_big_endian,
                  esbmct::expr2tc_source_value, esbmct::expr2tc_source_offset>;

class byte_update2t : public esbmct::expr<byte_update2t,
                                           esbmct::bool_big_endian,
                                           esbmct::expr2tc_source_value,
                                           esbmct::expr2tc_source_offset,
                                           esbmct::expr2tc_update_value>
{
public:
  byte_update2t(const type2tc &type, bool is_big_endian, const expr2tc &source,
                 const expr2tc &offset, const expr2tc &updateval)
    : esbmct::expr<byte_update2t, esbmct::bool_big_endian,
                 esbmct::expr2tc_source_value, esbmct::expr2tc_source_offset,
                 esbmct::expr2tc_update_value>
      (type, byte_update_id, is_big_endian, source, offset, updateval) {}
  byte_update2t(const byte_update2t &ref)
    : esbmct::expr<byte_update2t, esbmct::bool_big_endian,
                 esbmct::expr2tc_source_value, esbmct::expr2tc_source_offset,
                 esbmct::expr2tc_update_value>
      (ref) {}
};
template class esbmct::expr<byte_update2t, esbmct::bool_big_endian,
                  esbmct::expr2tc_source_value, esbmct::expr2tc_source_offset,
                  esbmct::expr2tc_update_value>;

class with2t : public esbmct::expr<with2t, esbmct::expr2tc_source_value,
                                         esbmct::expr2tc_update_field,
                                         esbmct::expr2tc_update_value>
{
public:
  with2t(const type2tc &type, const expr2tc &source, const expr2tc &field,
         const expr2tc &value)
    : esbmct::expr<with2t, esbmct::expr2tc_source_value,
                         esbmct::expr2tc_update_field,
                         esbmct::expr2tc_update_value>
      (type, with_id, source, field, value) {}
  with2t(const with2t &ref)
    : esbmct::expr<with2t, esbmct::expr2tc_source_value,
                         esbmct::expr2tc_update_field,
                         esbmct::expr2tc_update_value> (ref) {}
};
template class esbmct::expr<with2t, esbmct::expr2tc_source_value,
                                  esbmct::expr2tc_update_field,
                                  esbmct::expr2tc_update_value>;

class member2t : public esbmct::expr<member2t, esbmct::expr2tc_source_value,
                                             esbmct::irepidt_member>
{
public:
  member2t(const type2tc &type, const expr2tc &source, const irep_idt &memb)
    : esbmct::expr<member2t, esbmct::expr2tc_source_value, esbmct::irepidt_member>
      (type, member_id, source, memb) {}
  member2t(const member2t &ref)
    : esbmct::expr<member2t, esbmct::expr2tc_source_value, esbmct::irepidt_member>
      (ref) {}
};
template class esbmct::expr<member2t, esbmct::expr2tc_source_value,
                                    esbmct::irepidt_member>;

class index2t : public esbmct::expr<index2t, esbmct::expr2tc_source_value,
                                           esbmct::expr2tc_index>
{
public:
  index2t(const type2tc &type, const expr2tc &source, const expr2tc &index)
    : esbmct::expr<index2t, esbmct::expr2tc_source_value, esbmct::expr2tc_index>
      (type, index_id, source, index) {}
  index2t(const index2t &ref)
    : esbmct::expr<index2t, esbmct::expr2tc_source_value, esbmct::expr2tc_index>
      (ref) {}
};
template class esbmct::expr<index2t, esbmct::expr2tc_source_value,
                                   esbmct::expr2tc_index>;

class zero_string2t : public esbmct::expr<zero_string2t, esbmct::expr2tc_string>
{
public:
  zero_string2t(const expr2tc &string)
    : esbmct::expr<zero_string2t, esbmct::expr2tc_string>
      (type_pool.get_bool(), zero_string_id, string) {}
  zero_string2t(const zero_string2t &ref)
    : esbmct::expr<zero_string2t, esbmct::expr2tc_string>
      (ref) {}
};
template class esbmct::expr<zero_string2t, esbmct::expr2tc_string>;

class zero_length_string2t : public esbmct::expr<zero_length_string2t, esbmct::expr2tc_string>
{
public:
  zero_length_string2t(const expr2tc &string)
    : esbmct::expr<zero_length_string2t, esbmct::expr2tc_string>
      (type_pool.get_bool(), zero_length_string_id, string) {}
  zero_length_string2t(const zero_length_string2t &ref)
    : esbmct::expr<zero_length_string2t, esbmct::expr2tc_string>
      (ref) {}
};
template class esbmct::expr<zero_length_string2t, esbmct::expr2tc_string>;

class isnan2t : public esbmct::expr<isnan2t, esbmct::expr2tc_value>
{
public:
  isnan2t(const expr2tc &value)
    : esbmct::expr<isnan2t, esbmct::expr2tc_value>
      (type_pool.get_bool(), isnan_id, value) {}
  isnan2t(const isnan2t &ref)
    : esbmct::expr<isnan2t, esbmct::expr2tc_value>
      (ref) {}
};
template class esbmct::expr<isnan2t, esbmct::expr2tc_value>;

/** Check whether operand overflows. Operand must be either add, subtract,
 *  or multiply. XXXjmorse - in the future we should ensure the type of the
 *  operand is the expected type result of the operation. That way we can tell
 *  whether to do a signed or unsigned over/underflow test. */

class overflow2t : public esbmct::expr<overflow2t, esbmct::expr2tc_operand>
{
public:
  overflow2t(const expr2tc &operand)
    : esbmct::expr<overflow2t, esbmct::expr2tc_operand>
      (type_pool.get_bool(), overflow_id, operand) {}
  overflow2t(const overflow2t &ref)
    : esbmct::expr<overflow2t, esbmct::expr2tc_operand>
      (ref) {}
};
template class esbmct::expr<overflow2t, esbmct::expr2tc_operand>;

class overflow_cast2t : public esbmct::expr<overflow_cast2t,
                                          esbmct::uint_bits,
                                          esbmct::expr2tc_operand>
{
public:
  overflow_cast2t(const expr2tc &operand, unsigned int bits)
    : esbmct::expr<overflow_cast2t, esbmct::uint_bits, esbmct::expr2tc_operand>
      (type_pool.get_bool(), overflow_cast_id, bits, operand) {}
  overflow_cast2t(const overflow_cast2t &ref)
    : esbmct::expr<overflow_cast2t, esbmct::uint_bits, esbmct::expr2tc_operand>
      (ref) {}
};
template class esbmct::expr<overflow_cast2t, esbmct::uint_bits,
                                           esbmct::expr2tc_operand>;

class overflow_neg2t : public esbmct::expr<overflow_neg2t,
                                         esbmct::expr2tc_operand>
{
public:
  overflow_neg2t(const expr2tc &operand)
    : esbmct::expr<overflow_neg2t, esbmct::expr2tc_operand>
      (type_pool.get_bool(), overflow_neg_id, operand) {}
  overflow_neg2t(const overflow_neg2t &ref)
    : esbmct::expr<overflow_neg2t, esbmct::expr2tc_operand>
      (ref) {}
};
template class esbmct::expr<overflow_neg2t, esbmct::expr2tc_operand>;

inline bool operator==(boost::shared_ptr<type2t> const & a, boost::shared_ptr<type2t> const & b)
{
  return (*a.get() == *b.get());
}

inline bool operator!=(boost::shared_ptr<type2t> const & a, boost::shared_ptr<type2t> const & b)
{
  return !(a == b);
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

struct type2_hash
{
  size_t operator()(const type2tc &ref) const { return ref->crc(); }
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

inline bool is_structure_type(const type2tc &t)
{
  return t->type_id == type2t::struct_id || t->type_id == type2t::union_id;
}

inline const std::vector<type2tc> &get_structure_members(const type2tc &someval)
{
  if (is_struct_type(someval)) {
    const struct_type2t &v = static_cast<const struct_type2t &>(*someval.get());
    return v.members;
  } else {
    assert(is_union_type(someval));
    const union_type2t &v = static_cast<const union_type2t &>(*someval.get());
    return v.members;
  }
}

inline const std::vector<irep_idt> &get_structure_member_names(const type2tc &someval)
{
  if (is_struct_type(someval)) {
    const struct_type2t &v = static_cast<const struct_type2t &>(*someval.get());
    return v.member_names;
  } else {
    assert(is_union_type(someval));
    const union_type2t &v = static_cast<const union_type2t &>(*someval.get());
    return v.member_names;
  }
}

inline const irep_idt &get_structure_name(const type2tc &someval)
{
  if (is_struct_type(someval)) {
    const struct_type2t &v = static_cast<const struct_type2t &>(*someval.get());
    return v.name;
  } else {
    assert(is_union_type(someval));
    const union_type2t &v = static_cast<const union_type2t &>(*someval.get());
    return v.name;
  }
}

template <class T, expr2t::expr_ids expid>
class irep_container : protected boost::shared_ptr<T>
{
  irep_container() : boost::shared_ptr<T>() {}

  template<class Y>
  explicit irep_container(Y *p) : boost::shared_ptr<T>(p)
    { assert(p->expr_id == expid); }

  irep_container(const irep_container &ref)
    : boost::shared_ptr<T>(ref) {}

  irep_container(const expr2tc &ref)
    : boost::shared_ptr<T>
      (boost::shared_polymorphic_cast<T, expr2t>(ref))
      { assert(this->get()->expr_id == expid); }

  irep_container &operator=(irep_container const &ref)
  {
    boost::shared_ptr<T>::operator=(ref);
    return *this;
  }

  template<class Y>
  irep_container & operator=(boost::shared_ptr<Y> const & r)
  {
    boost::shared_ptr<T>::operator=(r);
    return *this;
  }

  const T * operator-> () const // never throws
  {
    return boost::shared_ptr<T>::operator->();
  }

  const T * get() const // never throws
  {
    return boost::shared_ptr<T>::get();
  }

  T * operator-> () // never throws
  {
    detach();
    return boost::shared_ptr<T>::operator->();
  }

  T * get() // never throws
  {
    detach();
    return boost::shared_ptr<T>::get();
  }

  void detach(void)
  {
    if (this->use_count() == 1)
      return; // No point remunging oneself if we're the only user of the ptr.

    // Assign-operate ourself into containing a fresh copy of the data. This
    // creates a new reference counted object, and assigns it to ourself,
    // which causes the existing reference to be decremented.
    *this = irep_container(get()->clone());
    return;
  }
};

#endif /* _UTIL_IREP2_H_ */
