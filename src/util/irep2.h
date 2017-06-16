#ifndef IREP2_H_
#define IREP2_H_

/** @file irep2.h
 *  Classes and definitions for non-stringy internal representation.
 */

#include <big-int/bigint.hh>
#include <boost/bind/placeholders.hpp>
#include <boost/crc.hpp>
#include <boost/functional/hash_fwd.hpp>
#include <boost/fusion/include/equal_to.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/push_front.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/preprocessor/list/adt.hpp>
#include <boost/preprocessor/list/for_each.hpp>
#include <boost/shared_ptr.hpp>
#include <cstdarg>
#include <functional>
#include <util/config.h>
#include <util/crypto_hash.h>
#include <util/dstring.h>
#include <util/irep.h>
#include <vector>

// Ahead of time: a list of all expressions and types, in a preprocessing
// list, for enumerating later. Should avoid manually enumerating anywhere
// else.

#define ESBMC_LIST_OF_EXPRS BOOST_PP_LIST_CONS(constant_int,\
  BOOST_PP_LIST_CONS(constant_fixedbv,\
  BOOST_PP_LIST_CONS(constant_floatbv,\
  BOOST_PP_LIST_CONS(constant_bool,\
  BOOST_PP_LIST_CONS(constant_string,\
  BOOST_PP_LIST_CONS(constant_struct,\
  BOOST_PP_LIST_CONS(constant_union,\
  BOOST_PP_LIST_CONS(constant_array,\
  BOOST_PP_LIST_CONS(constant_array_of,\
  BOOST_PP_LIST_CONS(symbol,\
  BOOST_PP_LIST_CONS(typecast,\
  BOOST_PP_LIST_CONS(bitcast,\
  BOOST_PP_LIST_CONS(nearbyint,\
  BOOST_PP_LIST_CONS(if,\
  BOOST_PP_LIST_CONS(equality,\
  BOOST_PP_LIST_CONS(notequal,\
  BOOST_PP_LIST_CONS(lessthan,\
  BOOST_PP_LIST_CONS(greaterthan,\
  BOOST_PP_LIST_CONS(lessthanequal,\
  BOOST_PP_LIST_CONS(greaterthanequal,\
  BOOST_PP_LIST_CONS(not,\
  BOOST_PP_LIST_CONS(and,\
  BOOST_PP_LIST_CONS(or,\
  BOOST_PP_LIST_CONS(xor,\
  BOOST_PP_LIST_CONS(implies,\
  BOOST_PP_LIST_CONS(bitand,\
  BOOST_PP_LIST_CONS(bitor,\
  BOOST_PP_LIST_CONS(bitxor,\
  BOOST_PP_LIST_CONS(bitnand,\
  BOOST_PP_LIST_CONS(bitnor,\
  BOOST_PP_LIST_CONS(bitnxor,\
  BOOST_PP_LIST_CONS(bitnot,\
  BOOST_PP_LIST_CONS(lshr,\
  BOOST_PP_LIST_CONS(neg,\
  BOOST_PP_LIST_CONS(abs,\
  BOOST_PP_LIST_CONS(add,\
  BOOST_PP_LIST_CONS(sub,\
  BOOST_PP_LIST_CONS(mul,\
  BOOST_PP_LIST_CONS(div,\
  BOOST_PP_LIST_CONS(ieee_add,\
  BOOST_PP_LIST_CONS(ieee_sub,\
  BOOST_PP_LIST_CONS(ieee_mul,\
  BOOST_PP_LIST_CONS(ieee_div,\
  BOOST_PP_LIST_CONS(ieee_fma,\
  BOOST_PP_LIST_CONS(ieee_sqrt,\
  BOOST_PP_LIST_CONS(modulus,\
  BOOST_PP_LIST_CONS(shl,\
  BOOST_PP_LIST_CONS(ashr,\
  BOOST_PP_LIST_CONS(dynamic_object,\
  BOOST_PP_LIST_CONS(same_object,\
  BOOST_PP_LIST_CONS(pointer_offset,\
  BOOST_PP_LIST_CONS(pointer_object,\
  BOOST_PP_LIST_CONS(address_of,\
  BOOST_PP_LIST_CONS(byte_extract,\
  BOOST_PP_LIST_CONS(byte_update,\
  BOOST_PP_LIST_CONS(with,\
  BOOST_PP_LIST_CONS(member,\
  BOOST_PP_LIST_CONS(index,\
  BOOST_PP_LIST_CONS(isnan,\
  BOOST_PP_LIST_CONS(overflow,\
  BOOST_PP_LIST_CONS(overflow_cast,\
  BOOST_PP_LIST_CONS(overflow_neg,\
  BOOST_PP_LIST_CONS(unknown,\
  BOOST_PP_LIST_CONS(invalid,\
  BOOST_PP_LIST_CONS(null_object,\
  BOOST_PP_LIST_CONS(dereference,\
  BOOST_PP_LIST_CONS(valid_object,\
  BOOST_PP_LIST_CONS(deallocated_obj,\
  BOOST_PP_LIST_CONS(dynamic_size,\
  BOOST_PP_LIST_CONS(sideeffect,\
  BOOST_PP_LIST_CONS(code_block,\
  BOOST_PP_LIST_CONS(code_assign,\
  BOOST_PP_LIST_CONS(code_init,\
  BOOST_PP_LIST_CONS(code_decl,\
  BOOST_PP_LIST_CONS(code_printf,\
  BOOST_PP_LIST_CONS(code_expression,\
  BOOST_PP_LIST_CONS(code_return,\
  BOOST_PP_LIST_CONS(code_skip,\
  BOOST_PP_LIST_CONS(code_free,\
  BOOST_PP_LIST_CONS(code_goto,\
  BOOST_PP_LIST_CONS(object_descriptor,\
  BOOST_PP_LIST_CONS(code_function_call,\
  BOOST_PP_LIST_CONS(code_comma,\
  BOOST_PP_LIST_CONS(invalid_pointer,\
  BOOST_PP_LIST_CONS(code_asm,\
  BOOST_PP_LIST_CONS(code_cpp_del_array,\
  BOOST_PP_LIST_CONS(code_cpp_delete,\
  BOOST_PP_LIST_CONS(code_cpp_catch,\
  BOOST_PP_LIST_CONS(code_cpp_throw,\
  BOOST_PP_LIST_CONS(code_cpp_throw_decl,\
  BOOST_PP_LIST_CONS(code_cpp_throw_decl_end,\
  BOOST_PP_LIST_CONS(isinf,\
  BOOST_PP_LIST_CONS(isnormal,\
  BOOST_PP_LIST_CONS(isfinite,\
  BOOST_PP_LIST_CONS(signbit,\
  BOOST_PP_LIST_CONS(concat, BOOST_PP_LIST_NIL)))))))))))))))))))))))))))))))\
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

#define ESBMC_LIST_OF_TYPES BOOST_PP_LIST_CONS(bool,\
BOOST_PP_LIST_CONS(empty,\
BOOST_PP_LIST_CONS(symbol,\
BOOST_PP_LIST_CONS(struct,\
BOOST_PP_LIST_CONS(union,\
BOOST_PP_LIST_CONS(code,\
BOOST_PP_LIST_CONS(array,\
BOOST_PP_LIST_CONS(pointer,\
BOOST_PP_LIST_CONS(unsignedbv,\
BOOST_PP_LIST_CONS(signedbv,\
BOOST_PP_LIST_CONS(fixedbv,\
BOOST_PP_LIST_CONS(string,\
BOOST_PP_LIST_CONS(cpp_name, BOOST_PP_LIST_NIL)))))))))))))

// Even crazier forward decs,
namespace esbmct {
  template <typename ...Args> class expr2t_traits;
  typedef expr2t_traits<> expr2t_default_traits;
  template <typename ...Args> class type2t_traits;
  typedef type2t_traits<> type2t_default_traits;
}

class type2t;
class expr2t;
class constant_array2t;

/** Reference counted container for expr2t based classes.
 *  This class extends boost shared_ptr's to contain anything that's a subclass
 *  of expr2t. It provides several ways of accessing the contained pointer;
 *  crucially it ensures that the only way to get a non-const reference or
 *  pointer is via the get() method, which call the detach() method.
 *
 *  This exists to ensure that we honour the model set forth by the old string
 *  based internal representation - specifically, that if you performed a const
 *  operation on an irept (fetching data) then the contained piece of data
 *  could continue to be shared between numerous data structures, for example
 *  a piece of code could exist in a contextt, a namespacet, and a goto_programt
 *  and all would share the same contained data structure, preventing additional
 *  memory consumption.
 *
 *  If anything copied an irept from one of these place it'd also share that
 *  contained data; but if it made a modifying operation (add, set, or just
 *  taking a non-const reference the contained data,) then the detach() method
 *  would be called, which duplicated the contained item and let the current
 *  piece of code modify the duplicate copy, while all the other storage
 *  locations continued to share the original.
 *
 *  So yeah, that's what this class attempts to implement, via the medium of
 *  boosts shared_ptr.
 */
template <class T>
class irep_container : public std::shared_ptr<T>
{
public:
  irep_container() : std::shared_ptr<T>() {}

  template<class Y>
  explicit irep_container(Y *p) : std::shared_ptr<T>(p)
    { }

  template<class Y>
  explicit irep_container(const Y *p) : std::shared_ptr<T>(const_cast<Y *>(p))
    { }

  irep_container(const irep_container &ref)
    : std::shared_ptr<T>(ref) {}

  template <class Y>
  irep_container(const irep_container<Y> &ref)
    : std::shared_ptr<T>(static_cast<const std::shared_ptr<Y> &>(ref))
  {
    assert(dynamic_cast<const std::shared_ptr<T> &>(ref) != NULL);
  }

  irep_container &operator=(irep_container const &ref)
  {
    std::shared_ptr<T>::operator=(ref);
    return *this;
  }

  template<class Y>
  irep_container & operator=(std::shared_ptr<Y> const & r)
  {
    std::shared_ptr<T>::operator=(r);
    T *p = std::shared_ptr<T>::operator->();
    return *this;
  }

  template <class Y>
  irep_container &operator=(const irep_container<Y> &ref)
  {
    assert(dynamic_cast<const std::shared_ptr<T> &>(ref) != NULL);
    *this = std::static_pointer_cast<T, Y>
            (static_cast<const std::shared_ptr<Y> &>(ref));
    return *this;
  }

  const T &operator*() const
  {
    return *std::shared_ptr<T>::get();
  }

  const T * operator-> () const // never throws
  {
    return std::shared_ptr<T>::operator->();
  }

  const T * get() const // never throws
  {
    return std::shared_ptr<T>::get();
  }

  T * get() // never throws
  {
    detach();
    T *tmp = std::shared_ptr<T>::get();
    tmp->crc_val = 0;
    return tmp;
  }

  void detach(void)
  {
    if (this->use_count() == 1)
      return; // No point remunging oneself if we're the only user of the ptr.

    // Assign-operate ourself into containing a fresh copy of the data. This
    // creates a new reference counted object, and assigns it to ourself,
    // which causes the existing reference to be decremented.
    const T *foo = std::shared_ptr<T>::get();
    *this = foo->clone();
    return;
  }

  uint32_t crc(void) const
  {
    const T *foo = std::shared_ptr<T>::get();
    if (foo->crc_val != 0)
      return foo->crc_val;

    foo->do_crc(0);
    return foo->crc_val;
  }
};

typedef irep_container<type2t> type2tc;
typedef irep_container<expr2t> expr2tc;

typedef std::pair<std::string,std::string> member_entryt;
typedef std::list<member_entryt> list_of_memberst;

/** Base class for all types.
 *  Contains only a type identifier enumeration - for some types (such as bool,
 *  or empty,) there's no need for any significant amount of data to be stored.
 */
class type2t
{
public:
  /** Enumeration identifying each sort of type. */
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
    floatbv_id,
    string_id,
    cpp_name_id,
    end_type_id
  };

  /* Define default traits */
  typedef typename esbmct::type2t_default_traits traits;

  /** Symbolic type exception class.
   *  To be thrown when attempting to fetch the width of a symbolic type, such
   *  as empty or code. Caller will have to worry about what to do about that.
   */
  class symbolic_type_excp {
  };

  typedef std::function<void (const type2tc &t)> const_subtype_delegate;
  typedef std::function<void (type2tc &t)> subtype_delegate;

protected:
  /** Primary constructor.
   *  @param id Type ID of type being constructed
   */
  type2t(type_ids id);

  /** Copy constructor */
  type2t(const type2t &ref);

  virtual void foreach_subtype_impl_const(const_subtype_delegate &t) const = 0;
  virtual void foreach_subtype_impl(subtype_delegate &t) = 0;

public:
  // Provide base / container types for some templates stuck on top:
  typedef type2tc container_type;
  typedef type2t base_type;

  virtual ~type2t() { };

  /** Fetch bit width of this type.
   *  For a particular type, calculate its size in a bit representation of
   *  itself. May throw various exceptions depending on whether this operation
   *  is viable - for example, for symbol types, infinite sized or dynamically
   *  sized arrays.
   *
   *  Note that the bit width is _not_ the same as the ansi-c byte model
   *  representation of this type.
   *
   *  @throws symbolic_type_excp
   *  @throws array_type2t::inf_sized_array_excp
   *  @throws array_type2t::dyn_sized_array_excp
   *  @return Size of types byte representation, in bits
   */
  virtual unsigned int get_width(void) const = 0;

  /* These are all self explanatory */
  bool operator==(const type2t &ref) const;
  bool operator!=(const type2t &ref) const;
  bool operator<(const type2t &ref) const;

  /** Produce a string representation of type.
   *  Takes body of the current type and produces a human readable
   *  representation. Similar to the string-irept's pretty method, although a
   *  different format.
   *  @param indent Number of spaces to indent lines by in the output
   *  @return String obj containing representation of this object
   */
  std::string pretty(unsigned int indent = 0) const;

  /** Dump object string representation to stdout.
   *  This take the output of the pretty method, and dumps it to stdout. To be
   *  used for debugging and when single stepping in gdb.
   *  @see pretty
   */
  void dump(void) const;

  /** Produce a checksum/hash of the current object.
   *  Takes current object and produces a lossy digest of it. Originally used
   *  crc32, now uses a more hacky but faster hash function. For use in hash
   *  objects.
   *  @see do_crc
   *  @return Digest of the current type.
   */
  uint32_t crc(void) const;

  /** Perform checked invocation of cmp method.
   *  Takes reference to another type - if they have the same type id, invoke
   *  the cmp function and return its result. Otherwise, return false. Using
   *  this method ensures thatthe implementer of cmp knows the reference it
   *  operates on is on the same type as itself.
   *  @param ref Reference to type to compare this object against
   *  @return True if types are the same, false otherwise.
   */
  bool cmpchecked(const type2t &ref) const;

  /** Perform checked invocation of lt method.
   *  Identical to cmpchecked, except with the lt method.
   *  @see cmpchecked
   *  @param ref Reference to type to measure this against.
   *  @return 0 if types are the same, 1 if this > ref, -1 if ref > this.
   */
  int ltchecked(const type2t &ref) const;

  /** Virtual method to compare two types.
   *  To be overridden by an extending type; assumes that itself and the
   *  parameter are of the same extended type. Call via cmpchecked.
   *  @see cmpchecked
   *  @param ref Reference to (same class of) type to compare against
   *  @return True if types match, false otherwise
   */
  virtual bool cmp(const type2t &ref) const = 0;

  /** Virtual method to compare two types.
   *  To be overridden by an extending type; assumes that itself and the
   *  parameter are of the same extended type. Call via cmpchecked.
   *  @see cmpchecked
   *  @param ref Reference to (same class of) type to compare against
   *  @return 0 if types are the same, 1 if this > ref, -1 if ref > this.
   */
  virtual int lt(const type2t &ref) const;

  /** Extract a list of members from type as strings.
   *  Produces a list of pairs, mapping a member name to a string value. Used
   *  in the body of the pretty method.
   *  @see pretty
   *  @param indent Number of spaces to indent output strings with, if multiline
   *  @return list of name:value pairs.
   */
  virtual list_of_memberst tostring(unsigned int indent) const = 0;

  /** Perform crc operation accumulating into parameter.
   *  Performs the operation of the crc method, but overridden to be specific to
   *  a particular type. Accumulates data into the hash object parameter.
   *  @see cmp
   *  @param seed Hash to accumulate hash data into.
   *  @return Hash value
   */
  virtual size_t do_crc(size_t seed) const;

  /** Perform hash operation accumulating into parameter.
   *  Feeds data as appropriate to the type of the expression into the
   *  parameter, to be hashed. Like crc and do_crc, but for some other kind
   *  of hash scenario.
   *  @see cmp
   *  @see crc
   *  @see do_crc
   *  @param hash Object to accumulate hash data into.
   */
  virtual void hash(crypto_hash &hash) const;

  /** Clone method. Self explanatory.
   *  @return New container, containing a duplicate of this object.
   */
  virtual type2tc clone(void) const = 0;

  // Please see the equivalent methods in expr2t for documentation
  template <typename T>
  void foreach_subtype(T &&t) const
  {
    const_subtype_delegate wrapped(std::cref(t));
    foreach_subtype_impl_const(wrapped);
  }

  template <typename T>
  void Foreach_subtype(T &&t)
  {
    subtype_delegate wrapped(std::ref(t));
    foreach_subtype_impl(wrapped);
  }

  /** Instance of type_ids recording this types type. */
  // XXX XXX XXX this should be const
  type_ids type_id;

  mutable size_t crc_val;
};

/** Fetch identifying name for a type.
 *  I.E., this is the class of the type, what you'd get if you called type.id()
 *  with the old stringy irep. Ideally this should be a class method, but as it
 *  was added as a hack I haven't got round to it yet.
 *  @param type Type to fetch identifier for
 *  @return String containing name of type class.
 */
std::string get_type_id(const type2t &type);

/** Fetch identifying name for a type.
 *  Just passes through to type2t accepting function with the same name.
 *  @param type Type to fetch identifier for
 *  @return String containing name of type class.
 */
static inline std::string get_type_id(const type2tc &type)
{
  return get_type_id(*type);
}

/** Base class for all expressions.
 *  In this base, contains an expression id used for distinguishing different
 *  classes of expr, in addition we have a type as all exprs should have types.
 */
class expr2t;
class expr2t : public std::enable_shared_from_this<expr2t>
{
public:
  /** Enumeration identifying each sort of expr.
   */
  enum expr_ids {
    // Boost preprocessor magic: enumerate over each expression and pump out
    // a foo_id enum element. See list of ireps at top of file.
#define _ESBMC_IREP2_EXPRID_ENUM(r, data, elem) BOOST_PP_CAT(elem,_id),
BOOST_PP_LIST_FOR_EACH(_ESBMC_IREP2_EXPRID_ENUM, foo, ESBMC_LIST_OF_EXPRS)
    end_expr_id
  };

  /** Type for list of constant expr operands */
  typedef std::list<const expr2tc*> expr_operands;
  /** Type for list of non-constant expr operands */
  typedef std::list<expr2tc*> Expr_operands;

  typedef std::function<void (const expr2tc &expr)> const_op_delegate;
  typedef std::function<void (expr2tc &expr)> op_delegate;

protected:
  /** Primary constructor.
   *  @param type Type of this new expr
   *  @param id Class identifier for this new expr
   */
  expr2t(const type2tc type, expr_ids id);
  /** Copy constructor */
  expr2t(const expr2t &ref);

  virtual void foreach_operand_impl_const(const_op_delegate &expr) const = 0;
  virtual void foreach_operand_impl(op_delegate &expr) = 0;

public:
  // Provide base / container types for some templates stuck on top:
  typedef expr2tc container_type;
  typedef expr2t base_type;
  // Also provide base traits
  typedef esbmct::expr2t_default_traits traits;

  virtual ~expr2t() { };

  /** Clone method. Self explanatory. */
  virtual expr2tc clone(void) const = 0;

  /* These are all self explanatory */
  bool operator==(const expr2t &ref) const;
  bool operator<(const expr2t &ref) const;
  bool operator!=(const expr2t &ref) const;

  /** Perform type-checked call to lt method.
   *  Checks that this object and the one we're comparing against have the same
   *  expr class, so that the lt method can assume it's working on objects of
   *  the same type.
   *  @see type2t::ltchecked
   *  @param ref Expression object we're comparing this object against.
   *  @return 0 If exprs are the same, 1 if this > ref, -1 if ref > this.
   */
  int ltchecked(const expr2t &ref) const;

  /** Produce textual representation of this expr.
   *  Like the stringy-irep's pretty method, this takes the current object and
   *  produces a textual representation that can be read by a human to
   *  understand what's going on.
   *  @param indent Number of spaces to indent the output string lines by
   *  @return String object containing textual expr representation.
   */
  std::string pretty(unsigned int indent = 0) const;

  /** Calculate number of exprs descending from this one.
   *  For statistics collection - calculates the number of expressions that
   *  make up this particular expression (i.e., count however many expr2tc's you
   *  can reach from this expr).
   *  @return Number of expr2tc's reachable from this node.
   */
  unsigned long num_nodes(void) const;

  /** Calculate max depth of exprs from this point.
   *  Looks at all sub-exprs of this expr, and calculates the longest chain one
   *  can descend before there are no more. Useful for statistics about the
   *  exprs we're dealing with.
   *  @return Number of expr2tc's reachable from this node.
   */
  unsigned long depth(void) const;

  /** Write textual representation of this object to stdout.
   *  For use in debugging - dumps the output of the pretty method to stdout.
   *  Can either be used in portion of code, or more commonly called from gdb.
   */
  void dump(void) const;

  /** Calculate a hash/digest of the current expr.
   *  For use in hash data structures; used to be a crc32, but is now a 16 bit
   *  hash function generated by myself to be fast. May not have nice
   *  distribution properties, but is at least fast.
   *  @return Hash value of this expr
   */
  uint32_t crc(void) const;

  /** Perform comparison operation between this and another expr.
   *  Overridden by subclasses of expr2t to compare different members of this
   *  and the passed in object. Assumes that the passed in object is the same
   *  class type as this; Should be called via operator==, which will do that
   *  check automagically.
   *  @see type2t::cmp
   *  @param ref Expr object to compare this against
   *  @return True if objects are the same; false otherwise.
   */
  virtual bool cmp(const expr2t &ref) const;

  /** Compare two expr objects.
   *  Overridden by subclasses - takes two expr objects (this and ref) of the
   *  same type, and compares them, in the same manner as memcmp. The assumption
   *  that the objects are of the same type means lt should be called via
   *  ltchecked to check for different expr types.
   *  @see type2t::lt
   *  @param ref Expr object to compare this against
   *  @return 0 If exprs are the same, 1 if this > ref, -1 if ref > this.
   */
  virtual int lt(const expr2t &ref) const;

  /** Convert fields of subclasses to a string representation.
   *  Used internally by the pretty method - creates a list of pairs
   *  representing the fields in the subclass. Each pair is a pair of strings
   *  of the form fieldname : value. The value may be multiline, in which case
   *  the new line will have at least indent number of indenting spaces.
   *  @param indent Number of spaces to indent multiline output by
   *  @return list of string pairs, of form fieldname:value
   */
  virtual list_of_memberst tostring(unsigned int indent) const = 0;

  /** Perform digest/hash function on expr object.
   *  Takes all fields in this exprs and adds them to the passed in hash object
   *  to compute an expression-hash. Overridden by subclasses.
   *  @param seed Hash to accumulate expression data into.
   *  @return Hash value
   */
  virtual size_t do_crc(size_t seed) const;

  /** Perform hash operation accumulating into parameter.
   *  Feeds data as appropriate to the type of the expression into the
   *  parameter, to be hashed. Like crc and do_crc, but for some other kind
   *  of hash scenario.
   *  @see cmp
   *  @see crc
   *  @see do_crc
   *  @param hash Object to accumulate hash data into.
   */
  virtual void hash(crypto_hash &hash) const;

  /** Fetch a sub-operand.
   *  These can come out of any field that is an expr2tc, or contains them.
   *  No particular numbering order is promised.
   */
  virtual const expr2tc *get_sub_expr(unsigned int idx) const = 0 ;

  /** Fetch a sub-operand. Non-const version.
   *  These can come out of any field that is an expr2tc, or contains them.
   *  No particular numbering order is promised.
   */
  virtual expr2tc *get_sub_expr_nc(unsigned int idx) = 0;

  /** Count the number of sub-exprs there are.
   */
  virtual unsigned int get_num_sub_exprs(void) const = 0 ;

  /** Simplify an expression.
   *  Similar to simplification in the string-based irep, this generates an
   *  expression with any calculations or operations that can be simplified,
   *  simplified. In contrast to the old form though, this creates a new expr
   *  if something gets simplified, just to make it clear exactly what's
   *  going on.
   *  @return Either a nil expr (null pointer contents) if nothing could be
   *          simplified or a simplified expression.
   */
  expr2tc simplify(void) const;

  /** expr-specific simplification methods.
   *  By default, an expression can't be simplified, and this method returns
   *  a nil expression to show that. However if simplification is possible, the
   *  subclass overrides this and if it can simplify its operands, returns a
   *  new simplified expression. It should attempt to modify itself (it's
   *  const).
   *
   *  If simplification failed the first time around, the simplify method will
   *  simplify this expressions individual operands,
   *  and will then call an expr with the simplified operands to see if it's now
   *  become simplifiable. This call occurs whether or not any operands were
   *  actually simplified, see below.
   *
   *  The 'second' parameter can be used to avoid invoking expensive attempts
   *  to simplify an expression more than once - on the first call to
   *  do_simplify this parameter will be false, then on the second it's be true,
   *  allowing method implementation to save the expensive stuff until all of
   *  its operands have certainly been simplified.
   *
   *  Currently simplification does some things that it shouldn't: pointer
   *  arithmetic for example. I'm not sure where this can be relocated to
   *  though.
   *  @param second Whether this is the second call to do_simplify on this obj
   *  @return expr2tc A nil expression if no simplifcation could occur, or a new
   *          simplified object if it can.
   */
  virtual expr2tc do_simplify(bool second = false) const;

  /** Indirect, abstract operand iteration.
   *
   *  Provide a lambda-based accessor equivalent to the forall_operands2 macro
   *  where anonymous code (actually a delegate?) gets run over each operand
   *  expression. Because the full type of the expression isn't known by the
   *  caller, and each delegate is it's own type, we need to wrap it in a
   *  std::function before funneling it through a virtual function.
   *
   *  For the purpose of this method, an operand is another instance of an
   *  expr2tc. This means the delegate will be called on any expr2tc field of
   *  the expression, in the order they appear in the traits. For a vector of
   *  expressions, the delegate will be called for each element, in order.
   *
   *  The uncapitalized version is const; the capitalized version is non-const
   *  (and so one needs to .get() a mutable expr2t pointer when calling). When
   *  modifying operands, preserving type correctness is imperative.
   *
   *  @param t A delegate to be called for each expression operand; must have
   *           a type of void f(const expr2tc &)
   */
  template <typename T>
  void foreach_operand(T &&t) const
  {
    const_op_delegate wrapped(std::cref(t));
    foreach_operand_impl_const(wrapped);
  }

  template <typename T>
  void Foreach_operand(T &&t)
  {
    op_delegate wrapped(std::ref(t));
    foreach_operand_impl(wrapped);
  }

  /** Instance of expr_ids recording tihs exprs type. */
  const expr_ids expr_id;

  /** Type of this expr. All exprs have a type. */
  type2tc type;

  mutable size_t crc_val;
};

inline bool is_nil_expr(const expr2tc &exp)
{
  if (exp.get() == nullptr)
    return true;
  return false;
}

inline bool is_nil_type(const type2tc &t)
{
  if (t.get() == nullptr)
    return true;
  return false;
}

// For boost multi-index hashing,
inline std::size_t
hash_value(const expr2tc &expr)
{
  return expr.crc();
}

/** Fetch string identifier for an expression.
 *  Returns the class name of the expr passed in - this is equivalent to the
 *  result of expr.id() in old stringy irep. Should ideally be a method of
 *  expr2t, but haven't got around to moving it yet.
 *  @param expr Expression to operate upon
 *  @return String containing class name of expression.
 */
std::string get_expr_id(const expr2t &expr);

/** Fetch string identifier for an expression.
 *  Like the expr2t equivalent with the same name, but de-ensapculates an
 *  expr2tc.
 */
static inline std::string get_expr_id(const expr2tc &expr)
{
  return get_expr_id(*expr);
}

/** Template for providing templated methods to irep classes (type2t/expr2t).
 *
 *  What this does: we give irep_methods2 a type trait record that contains
 *  a boost::mpl::vector, the elements of which describe each field in the
 *  class we're operating on. For each field we get:
 *
 *    - The type of the field
 *    - The class that field is part of
 *    - A pointer offset to that field.
 *
 *  What this means, is that we can @a type @a generically access a member
 *  of a class from within the template, without knowing what type it is,
 *  what its name is, or even what type contains it.
 *
 *  We can then use that to make all the boring methods of ireps type
 *  generic too. For example: we can make the comparision method by accessing
 *  each field in the class we're dealing with, passing them to another
 *  function to do the comparison (with the type resolved by templates or
 *  via overloading), and then inspecting the output of that.
 *
 *  In fact, we can make type generic implementations of all the following
 *  methods in expr2t: clone, tostring, cmp, lt, do_crc, hash.
 *  Similar methods, minus the operands, can be made generic in type2t.
 *
 *  So, that's what these templates provide; an irep class can be made by
 *  inheriting from this template, telling it what class it'll end up with,
 *  and what to subclass from, and what the fields in the class being derived
 *  from look like. This means we can construct a type hierarchy with whatever
 *  inheretence we like and whatever fields we like, then latch irep_methods2
 *  on top of that to implement all the anoying boring boilerplate code.
 *
 *  ----
 *
 *  In addition, we also define container types for each irep, which is
 *  essentially a type-safeish wrapper around a std::shared_ptr (i.e.,
 *  reference counter). One can create a new irep with syntax such as:
 *
 *    foo2tc bar(type, operand1, operand2);
 *
 *  As well as copy-constructing a container around an expr to make it type
 *  specific:
 *
 *    expr2tc foo = something();
 *    foo2tc bar(foo);
 *
 *  Assertions in the construction will ensure that the expression is in fact
 *  of type foo2t. One can transparently access the irep fields through
 *  dereference, such as:
 *
 *    bar->operand1 = 0;
 *
 *  This all replicates the CBMC expression situation, but with the addition
 *  of types.
 *
 *  ----
 *
 *  Problems: there's an ambiguity between the construction of some new ireps,
 *  and the downcasting from one type to another. If one were to say:
 *
 *    not2tc foo(someotherexpr);
 *
 *  Are we constructing a new "not" expression, the inversion of someotherexpr,
 *  or downcasting it to a not2t reference? Currently it's configurable with
 *  some traits hacks, but the ambiguity is alas something that has to be lived
 *  with. All similar ireps are configured to always construct.
 *
 *  (The required traits hacks need cleaning up too).
 */
namespace esbmct {

  /** Maximum number of fields to support in expr2t subclasses. This value
   *  controls the types of any arrays that need to consider the number of
   *  fields.
   *  I've yet to find a way of making this play nice with the new variardic
   *  way of defining ireps. */
  const unsigned int num_type_fields = 6;

  // Dummy type tag - exists to be an arbitary, local class, for use in some
  // templates. See below.
  class dummy_type_tag {
  public:
    typedef int type;
  };

  /** Record for properties of an irep field.
   *  This type records, for any particular field:
   *    * It's type
   *    * The class that it's a member of
   *    * A class pointer to this field
   *  The aim being that we have enough information about the field to
   *  manipulate it without any further traits. */
  template <typename R, typename C, R C::* v>
    class field_traits
  {
  public:
    typedef R result_type;
    typedef C source_class;
    typedef R C::* membr_ptr;
    static constexpr membr_ptr value = v;
  };

  // Field traits specialization for const fields, i.e. expr_id. These become
  // landmines for future mutable methods, i.e. get_sub_expr, which may get
  // it's consts mixed up.
  template <typename R, typename C, R C::* v>
    class field_traits<const R, C, v>
  {
  public:
    typedef R result_type;
    typedef C source_class;
    typedef const R C::* membr_ptr;
    static constexpr membr_ptr value = v;
  };

  template <typename R, typename C, R C::* v>
  constexpr typename field_traits<R, C, v>::membr_ptr field_traits<R, C, v>::value;

  /** Trait class for type2t ireps.
   *  This takes a list of field traits and puts it in a vector, with the record
   *  for the type_id field (common to all type2t's) put that the front. */
  template <typename ...Args>
    class type2t_traits
  {
  public:
    typedef field_traits<type2t::type_ids, type2t, &type2t::type_id> type_id_field;
    typedef typename boost::mpl::push_front<boost::mpl::vector<Args...>, type_id_field>::type fields;
    static constexpr bool always_construct = false;
    typedef type2t base2t;

    template <typename derived>
    static irep_container<base2t> make_contained(typename Args::result_type...);
  };

  /** Trait class for expr2t ireps.
   *  This takes a list of field traits and puts it in a vector, with the record
   *  for the expr_id field (common to all expr2t's) put that the front. Records
   *  some additional flags about the usage of the expression -- specifically
   *  what a unary constructor will do (@see something2tc::something2tc) */
  template <typename ...Args>
    class expr2t_traits
  {
  public:
    typedef field_traits<const expr2t::expr_ids, expr2t, &expr2t::expr_id> expr_id_field;
    typedef field_traits<type2tc, expr2t, &expr2t::type> type_field;
    typedef typename boost::mpl::push_front<typename boost::mpl::push_front<boost::mpl::vector<Args...>, type_field>::type, expr_id_field>::type fields;
    static constexpr bool always_construct = false;
    static constexpr unsigned int num_fields = boost::mpl::size<fields>::type::value;
    typedef expr2t base2t;

    // Note addition of type2tc...
    template <typename derived>
    static irep_container<base2t> make_contained(const type2tc &, typename Args::result_type...);
  };

  // "Specialisation" for expr kinds that don't take a type, like boolean
  // typed exprs. Should actually become a more structured expr2t_traits
  // that can be specialised in this way, at a later date. Might want to
  // move the presumed type down to the _data class at that time too.
  template <typename ...Args>
    class expr2t_traits_notype
  {
  public:
    typedef field_traits<const expr2t::expr_ids, expr2t, &expr2t::expr_id> expr_id_field;
    typedef field_traits<type2tc, expr2t, &expr2t::type> type_field;
    typedef typename boost::mpl::push_front<typename boost::mpl::push_front<boost::mpl::vector<Args...>, type_field>::type, expr_id_field>::type fields;
    static constexpr bool always_construct = false;
    static constexpr unsigned int num_fields = boost::mpl::size<fields>::type::value;
    typedef expr2t base2t;

    template <typename derived>
    static irep_container<base2t> make_contained(typename Args::result_type...);
  };

  // Hack to force something2tc to always construct the traits' type, rather
  // that copy construct. Due to misery and ambiguity elsewhere.
  template <typename ...Args>
    class expr2t_traits_always_construct
  {
  public:
    typedef field_traits<const expr2t::expr_ids, expr2t, &expr2t::expr_id> expr_id_field;
    typedef typename boost::mpl::push_front<boost::mpl::vector<Args...>, expr_id_field>::type fields;
    static constexpr bool always_construct = true;
    static constexpr unsigned int num_fields = boost::mpl::size<fields>::type::value;
    typedef expr2t base2t;

    template <typename derived>
    static irep_container<base2t> make_contained(typename Args::result_type...);
  };

  // Declaration of irep and expr methods templates.
  template <class derived, class baseclass, typename traits, typename container, typename fields = typename traits::fields, typename enable = void>
    class irep_methods2;
  template <class derived, class baseclass, typename traits, typename container, typename fields = typename traits::fields, typename enable = void>
    class expr_methods2;
  template <class derived, class baseclass, typename traits, typename container, typename fields = typename traits::fields, typename enable = void>
    class type_methods2;

  /** Definition of irep methods template.
   *
   *  @param derived The inheritor class, like add2t
   *  @param baseclass Class containing fields for methods to be defined over
   *  @param traits Type traits for baseclass
   *
   *  A typical irep inheritance looks like this, descending from the base
   *  irep class to the most derived class:
   *
   *    b          Base class, such as type2t or expr2t
   *    d          Data class, containing storage fields for ireps
   *    m          Terminal methods class (see below)
   *    M
   *    M            Recursive chain of irep_methods2 classes. Each one
   *    M            implements methods for one field, and calls to a superclass
   *    M            to handle remaining fields
   *    M
   *    t          Top level class such as add2t
   *
   *  The effect is thus: one takes a base class containing storage fields,
   *  instantiate irep_methods2 on top of it which unrolls to one template
   *  instance per field (plus a specialized terminal when there are no more
   *  fields). Then, have the top level class inherit from the chain of
   *  irep_methods classes. This avoids the writing of certain boilerplate
   *  methods at the expense of writing type trait information.
   *
   *  Technically one could typedef the top level irep_methods class to be the
   *  top level class itself; however putting a 'cap' on it (as it were) avoids
   *  decades worth of template errors if a programmer uses the irep
   *  incorrectly.
   */
  template <class derived, class baseclass, typename traits, typename container, typename fields, typename enable>
    class irep_methods2 : public irep_methods2<derived, baseclass, traits, container, typename boost::mpl::pop_front<fields>::type>
  {
  public:
    typedef irep_methods2<derived, baseclass, traits, container, typename boost::mpl::pop_front<fields>::type> superclass;
    typedef container container2tc;
    typedef typename container::base_container base_container2tc;
    typedef typename baseclass::base_type base2t;

    template <typename ...Args> irep_methods2(const Args& ... args) : superclass(args...) { }

    // Copy constructor. Construct from derived ref rather than just
    // irep_methods2, because the template above will be able to directly
    // match a const derived &, and so the compiler won't cast it up to
    // const irep_methods2 & and call the copy constructor. Fix this by
    // defining a copy constructor that exactly matches the (only) use case.
    irep_methods2(const derived &ref) : superclass(ref) { }

    // Top level / public methods for this irep. These methods are virtual, set
    // up any relevant computation, and then call the recursive instances below
    // to perform the actual work over fields.
    base_container2tc clone(void) const;
    list_of_memberst tostring(unsigned int indent) const;
    bool cmp(const base2t &ref) const;
    int lt(const base2t &ref) const;
    size_t do_crc(size_t seed) const;
    void hash(crypto_hash &hash) const;

    static void build_python_class(const typename container::id_field_type id);

  protected:
    // Fetch the type information about the field we are concerned with out
    // of the current type trait we're working on.
    typedef typename boost::mpl::front<fields>::type::result_type cur_type;
    typedef typename boost::mpl::front<fields>::type::source_class base_class;
    typedef typename boost::mpl::front<fields>::type membr_ptr;

    // Recursive instances of boilerplate methods.
    void tostring_rec(unsigned int idx, list_of_memberst &vec, unsigned int indent) const;
    bool cmp_rec(const base2t &ref) const;
    int lt_rec(const base2t &ref) const;
    void do_crc_rec() const;
    void hash_rec(crypto_hash &hash) const;

    // These methods are specific to expressions rather than types, and are
    // placed here to avoid un-necessary recursion in expr_methods2.
    const expr2tc *get_sub_expr_rec(unsigned int cur_count, unsigned int desired) const;
    expr2tc *get_sub_expr_nc_rec(unsigned int cur_count, unsigned int desired);
    unsigned int get_num_sub_exprs_rec(void) const;

    void foreach_operand_impl_rec(expr2t::op_delegate &f);
    void foreach_operand_impl_const_rec(expr2t::const_op_delegate &f) const;

    // Similar story, but for type2tc
    void foreach_subtype_impl_rec(type2t::subtype_delegate &t);
    void foreach_subtype_impl_const_rec(type2t::const_subtype_delegate &t)const;

    template <typename T>
    static void build_python_class_rec(T &obj, unsigned int idx);
  };

  // Base instance of irep_methods2. This is a template specialization that
  // matches (via boost::enable_if) when the list of fields to operate on is
  // now empty. Finish up the remaining computation, if any.
  template <class derived, class baseclass, typename traits, typename container, typename fields>
    class irep_methods2<derived, baseclass, traits, container,
                        fields,
                        typename boost::enable_if<typename boost::mpl::empty<fields>::type>::type>
      : public baseclass
  {
  public:
    template <typename ...Args> irep_methods2(Args... args) : baseclass(args...) { }

    // Copy constructor. See note for non-specialized definition.
    irep_methods2(const derived &ref) : baseclass(ref) { }

  protected:
    typedef typename baseclass::container_type container2tc;
    typedef typename baseclass::base_type base2t;

    void tostring_rec(unsigned int idx, list_of_memberst &vec, unsigned int indent) const
    {
      (void)idx;
      (void)vec;
      (void)indent;
      return;
    }

    bool cmp_rec(const base2t &ref) const
    {
      // If it made it this far, we passed
      (void)ref;
      return true;
    }

    int lt_rec(const base2t &ref) const
    {
      // If it made it this far, we passed
      (void)ref;
      return 0;
    }

    void do_crc_rec() const
    {
      return;
    }

    void hash_rec(crypto_hash &hash) const
    {
      (void)hash;
      return;
    }

    const expr2tc *get_sub_expr_rec(unsigned int cur_idx, unsigned int desired) const
    {
      // No result, so desired must exceed the number of idx's
      assert(cur_idx >= desired);
      (void)cur_idx;
      (void)desired;
      return NULL;
    }

    expr2tc *get_sub_expr_nc_rec(unsigned int cur_idx, unsigned int desired)
    {
      // See above
      assert(cur_idx >= desired);
      (void)cur_idx;
      (void)desired;
      return NULL;
    }

    unsigned int get_num_sub_exprs_rec(void) const
    {
      return 0;
    }

    void foreach_operand_impl_rec(expr2t::op_delegate &f)
    {
      (void)f;
      return;
    }

    void foreach_operand_impl_const_rec(expr2t::const_op_delegate &f) const
    {
      (void)f;
      return;
    }

    void foreach_subtype_impl_rec(type2t::subtype_delegate &t)
    {
      (void)t;
      return;
    }

    void foreach_subtype_impl_const_rec(type2t::const_subtype_delegate &t) const
    {
      (void)t;
      return;
    }

    template <typename T>
    static void build_python_class_rec(T &obj, unsigned int idx)
    {
      (void)obj;
      (void)idx;
    }
  };

  /** Expression methods template for expr ireps.
   *  This class works on the same principle as @irep_methods2 but provides
   *  head methods for get_sub_expr and so forth, which are
   *  specific to expression ireps. The actual implementation of these methods
   *  are provided in irep_methods to avoid un-necessary recursion but are
   *  protected; here we provide the head methods publically to allow the
   *  programmer to call in.
   *  */
  template <class derived, class baseclass, typename traits, typename container, typename fields, typename enable>
    class expr_methods2 : public irep_methods2<derived, baseclass, traits, container, fields, enable>
  {
  public:
    typedef irep_methods2<derived, baseclass, traits, container, fields, enable> superclass;

    template <typename ...Args> expr_methods2(const Args&... args) : superclass(args...) { }

    // See notes on irep_methods2 copy constructor
    expr_methods2(const derived &ref) : superclass(ref) { }

    const expr2tc *get_sub_expr(unsigned int i) const;
    expr2tc *get_sub_expr_nc(unsigned int i);
    unsigned int get_num_sub_exprs(void) const;

    void foreach_operand_impl_const(expr2t::const_op_delegate &expr) const;
    void foreach_operand_impl(expr2t::op_delegate &expr);
  };

  /** Type methods template for type ireps.
   *  Like @expr_methods2, but for types. Also; written on the quick.
   *  */
  template <class derived, class baseclass, typename traits, typename container, typename fields, typename enable>
    class type_methods2 : public irep_methods2<derived, baseclass, traits, container, fields, enable>
  {
  public:
    typedef irep_methods2<derived, baseclass, traits, container, fields, enable> superclass;

    template <typename ...Args> type_methods2(const Args&... args) : superclass(args...) { }

    // See notes on irep_methods2 copy constructor
    type_methods2(const derived &ref) : superclass(ref) { }

    void foreach_subtype_impl_const(type2t::const_subtype_delegate &t) const;
    void foreach_subtype_impl(type2t::subtype_delegate &t);
  };

  // So that we can write such things as:
  //
  //   constant_int2tc bees(type, val);
  //
  // We need a class derived from expr2tc that takes the correct set of
  // constructor arguments, which means yet more template goo.
  template <class base, class contained, unsigned int expid, typename idtype, idtype base::*idfield, class superclass>
  class something2tc : public irep_container<base> {
    public:
      typedef irep_container<base> base2tc;
    // Blank initialization of a container class -> store NULL
    something2tc() : base2tc() { }

    // Initialize container from a non-type-committed container. Encode an
    // assertion that the type is what we expect.
    //
    // Don't do this though if this'll conflict with a later consructor though.
    // For example if we have not2tc, not2tc(expr) could be copying it or
    // constructing a new not2t irep. In the face of this ambiguity, pick the
    // latter, and the end user can worry about how to cast up to a not2tc.
    template <class arbitary = ::esbmct::dummy_type_tag>
    something2tc(const base2tc &init,
                 typename boost::lazy_disable_if<boost::mpl::bool_<superclass::traits::always_construct == true>, arbitary>::type* = NULL
                 ) : base2tc(init)
    {
      assert(init.get()->*idfield == expid);
    }

    // Allow construction too when we're handed a pointer to the (correctly
    // typed) base2t ptr. This is used by boost::python, and various bits of
    // code that create new ptrs and fling them into type2tcs.
    something2tc(contained *init) : base2tc(init)
    {
      assert(init != NULL); // Would already have fired right?
      assert(init->*idfield == expid);
    }

    const contained &operator*() const
    {
      return static_cast<const contained&>(*base2tc::get());
    }

    const contained * operator-> () const // never throws
    {
      return static_cast<const contained*>(base2tc::operator->());
    }

    const contained * get() const // never throws
    {
      return static_cast<const contained*>(base2tc::get());
    }

    contained * get() // never throws
    {
      base2tc::detach();
      return static_cast<contained*>(base2tc::get());
    }

    // Forward all constructors down to the contained type.
    template <typename ...Args>
    something2tc(Args... args) : base2tc(new contained(args...)) { }

    typedef irep_container<base> base_container;
    typedef idtype id_field_type;
  };

  // Boost doesn't have variadic vector templates, so convert to it.

  template <typename ...Args> class variadic_vector;

  template <typename T, typename ...Args>
  class variadic_vector<T, Args...>
  {
    typedef boost::mpl::push_back<variadic_vector<Args...>, T> type;
  };

  template <>
  class variadic_vector<>
  {
    typedef boost::mpl::vector<> type;
  };
}; // esbmct

// In global namespace: to get boost to recognize something2tc's as being a
// shared pointer type, we need to define get_pointer for it:

template <typename T1, typename T2, unsigned int T3, typename T4, T4 T1::*T5, typename T6>
T2* get_pointer(esbmct::something2tc<T1, T2, T3, T4, T5, T6> const& p) {
  return const_cast<T2*>(p.get());
}

inline bool operator==(const type2tc &a, const type2tc &b)
{
  // Handle nil ireps
  if (is_nil_type(a) && is_nil_type(b))
    return true;
  else if (is_nil_type(a) || is_nil_type(b))
    return false;
  else
    return (*a.get() == *b.get());
}

inline bool operator!=(const type2tc &a, const type2tc &b)
{
  return !(a == b);
}

inline bool operator<(const type2tc &a, const type2tc &b)
{
  if (is_nil_type(a)) // nil is lower than non-nil
    return !is_nil_type(b); // true if b is non-nil, so a is lower
  else if (is_nil_type(b))
    return false; // If b is nil, nothing can be lower
  else
    return (*a.get() < *b.get());
}

inline bool operator>(const type2tc &a, const type2tc &b)
{
  // We're greater if we neither less than or equal.
  // This costs more: but that's ok, because all conventional software uses
  // less-than comparisons for ordering
  return !(a < b) && (a != b);
}

inline bool operator==(const expr2tc& a, const expr2tc& b)
{
  if (is_nil_expr(a) && is_nil_expr(b))
    return true;
  else if (is_nil_expr(a) || is_nil_expr(b))
    return false;
  else
    return (*a.get() == *b.get());
}

inline bool operator!=(const expr2tc& a, const expr2tc& b)
{
  return !(a == b);
}

inline bool operator<(const expr2tc& a, const expr2tc& b)
{
  if (is_nil_expr(a)) // nil is lower than non-nil
    return !is_nil_expr(b); // true if b is non-nil, so a is lower
  else if (is_nil_expr(b))
    return false; // If b is nil, nothing can be lower
  else
    return (*a.get() < *b.get());
}

inline bool operator>(const expr2tc& a, const expr2tc& b)
{
  // We're greater if we neither less than or equal.
  // This costs more: but that's ok, because all conventional software uses
  // less-than comparisons for ordering
  return !(a < b) && (a != b);
}

inline std::ostream& operator<<(std::ostream &out, const expr2tc& a)
{
  out << a->pretty(0);
  return out;
}

struct irep2_hash
{
  size_t operator()(const expr2tc &ref) const { return ref.crc(); }
};

struct type2_hash
{
  size_t operator()(const type2tc &ref) const { return ref->crc(); }
};

#endif /* IREP2_H_ */
