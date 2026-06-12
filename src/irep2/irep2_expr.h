#ifndef IREP2_EXPR_H_
#define IREP2_EXPR_H_

#include <util/config.h>
#include <util/c_types.h>
#include <util/fixedbv.h>
#include <util/ieee_float.h>
#include <util/location.h>
#include <irep2/irep2_type.h>

// So - make some type definitions for the different types we're going to be
// working with. This is to avoid the repeated use of template names in later
// definitions. If you'd like to add another type - don't. Vast tracts of code
// only expect the types below, it's be extremely difficult to hack new ones in.

// Start of definitions for expressions. Forward decls
//
// Forward-declare a concrete <kind>2t class for every entry in the
// expr_kinds.inc manifest. The same manifest drives the expr_ids
// enum (in irep2.h) and the is_/to_/try_to_ predicate generators
// further down this file.
#define IREP2_EXPR(kind, pretty) class kind##2t;
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR

enum class constant_string_kindt
{
  DEFAULT, /* "" */
  WIDE,    /* L"" */
  UNICODE, /* u8"", u"" and U"" */
};

/** Symex renaming level for symbol2t.
 *
 * Symbolic execution rewrites a symbol into successively more specific
 * variants as it threads constraints through the SSA program:
 *
 *   - level0          — the raw symbol straight from the frontend, no
 *                       activation/SSA decoration applied yet.
 *   - level1          — annotated with the function activation record
 *                       (level1_num) and owning thread (thread_num); the
 *                       symbol refers to a particular per-thread, per-call
 *                       instance of a local.
 *   - level2          — additionally annotated with an SSA assignment
 *                       counter (level2_num, node_num); the symbol refers
 *                       to a specific value version of that local.
 *   - level1_global   — like level1, but for a globally-scoped symbol
 *                       (no activation record applies; it is shared
 *                       across functions).
 *   - level2_global   — like level2, but for a globally-scoped symbol.
 *
 * See src/goto-symex/renaming.cpp for the exact transitions. */
enum class symbol_renaming_level
{
  level0,
  level1,
  level2,
  level1_global,
  level2_global,
};

/** Debug-only consistency check for arith_2ops operands and result type.
 *  Called from add2t/sub2t/mul2t/div2t/modulus2t constructors; no-op in Release. */
void assert_arith_2ops_consistency(
  const type2tc &t,
  expr2t::expr_ids id,
  const expr2tc &v1,
  const expr2tc &v2);

/** Enumeration identifying each particular kind of side effect. */
enum class sideeffect_allockind
{
  malloc,
  realloc,
  alloca,
  cpp_new,
  cpp_new_arr,
  nondet,
  va_arg,
  printf2,
  function_call,
  preincrement,
  postincrement,
  predecrement,
  postdecrement,
  old_snapshot,               // For __ESBMC_old() in function contracts
  assigns_target,             // For __ESBMC_assigns() in function contracts
  statement_expression,       // GNU C ({ ... }) extension
  temporary_object,           // C++ temporary created inline (constructor)
  gcc_conditional_expression, // GNU C `a ?: b` (omitted middle operand)
  cpp_delete,                 // C++ `delete p` in expression position
  cpp_delete_array            // C++ `delete[] p` in expression position
};

/** Which member of the printf family a `code_printf2t` represents. */
enum class printf_kindt
{
  PRINTF,
  FPRINTF,
  DPRINTF,
  SPRINTF,
  VFPRINTF,
  SNPRINTF,
  VPRINTF,
  VSPRINTF,
  VSNPRINTF,
  ASPRINTF,
  VASPRINTF,
};

/** Maps the textual base_name of a printf-family symbol (e.g. "printf",
 *  "snprintf") onto a printf_kindt.  Aborts on an unknown name. */
printf_kindt printf_kind_from_name(const irep_idt &name);

#define irep_typedefs(basename)                                                \
  template <typename... Args>                                                  \
  inline expr2tc basename##2tc(Args && ...args)                                \
  {                                                                            \
    return make_irep<basename##2t>(std::forward<Args>(args)...);               \
  }

irep_typedefs(constant_int);
irep_typedefs(constant_fixedbv);
irep_typedefs(constant_floatbv);
irep_typedefs(constant_struct);
irep_typedefs(constant_union);
irep_typedefs(constant_array);
irep_typedefs(constant_vector);
irep_typedefs(constant_bool);
irep_typedefs(constant_array_of);
irep_typedefs(constant_string);
irep_typedefs(symbol);
irep_typedefs(nearbyint);
irep_typedefs(typecast);
irep_typedefs(bitcast);
irep_typedefs(if);
irep_typedefs(equality);
irep_typedefs(notequal);
irep_typedefs(lessthan);
irep_typedefs(greaterthan);
irep_typedefs(lessthanequal);
irep_typedefs(greaterthanequal);
irep_typedefs(cmp_three_way);
irep_typedefs(not );
irep_typedefs(and);
irep_typedefs(or);
irep_typedefs(xor);
irep_typedefs(implies);
irep_typedefs(bitand);
irep_typedefs(bitor);
irep_typedefs(bitxor);
irep_typedefs(lshr);
irep_typedefs(bitnot);
irep_typedefs(neg);
irep_typedefs(abs);
irep_typedefs(add);
irep_typedefs(sub);
irep_typedefs(mul);
irep_typedefs(div);
irep_typedefs(ieee_add);
irep_typedefs(ieee_sub);
irep_typedefs(ieee_mul);
irep_typedefs(ieee_div);
irep_typedefs(ieee_fma);
irep_typedefs(ieee_sqrt);
irep_typedefs(modulus);
irep_typedefs(shl);
irep_typedefs(ashr);
irep_typedefs(same_object);
irep_typedefs(pointer_offset);
irep_typedefs(pointer_object);
irep_typedefs(pointer_capability);
irep_typedefs(address_of);
irep_typedefs(byte_extract);
irep_typedefs(byte_update);
irep_typedefs(with);
irep_typedefs(member);
irep_typedefs(member_ref);
irep_typedefs(ptr_mem);
irep_typedefs(index);
irep_typedefs(isnan);
irep_typedefs(overflow);
irep_typedefs(overflow_cast);
irep_typedefs(overflow_neg);
irep_typedefs(unknown);
irep_typedefs(invalid);
irep_typedefs(null_object);
irep_typedefs(dynamic_object);
irep_typedefs(dereference);
irep_typedefs(valid_object);
irep_typedefs(races_check);
irep_typedefs(deallocated_obj);
irep_typedefs(dynamic_size);
irep_typedefs(sideeffect);
irep_typedefs(code_block);
irep_typedefs(code_assign);
irep_typedefs(code_decl);
irep_typedefs(code_dead);
irep_typedefs(code_printf);
irep_typedefs(code_expression);
irep_typedefs(code_return);
irep_typedefs(code_skip);
irep_typedefs(code_free);
irep_typedefs(code_goto);
irep_typedefs(object_descriptor);
irep_typedefs(code_function_call);
irep_typedefs(code_ifthenelse);
irep_typedefs(code_while);
irep_typedefs(code_dowhile);
irep_typedefs(code_for);
irep_typedefs(code_switch);
irep_typedefs(code_break);
irep_typedefs(code_continue);
irep_typedefs(code_label);
irep_typedefs(code_switch_case);
irep_typedefs(code_assert);
irep_typedefs(code_assume);
irep_typedefs(sideeffect_assign);
irep_typedefs(code_comma);
irep_typedefs(invalid_pointer);
irep_typedefs(code_asm);
irep_typedefs(code_cpp_del_array);
irep_typedefs(code_cpp_delete);
irep_typedefs(code_cpp_catch);
irep_typedefs(code_cpp_throw);
irep_typedefs(isinf);
irep_typedefs(isnormal);
irep_typedefs(isfinite);
irep_typedefs(signbit);
irep_typedefs(popcount);
irep_typedefs(bswap);
irep_typedefs(concat);
irep_typedefs(extract);
irep_typedefs(capability_base);
irep_typedefs(capability_top);
irep_typedefs(forall);
irep_typedefs(exists);
irep_typedefs(isinstance);
irep_typedefs(hasattr);
irep_typedefs(isnone);
irep_typedefs(new_object);

class exists2t : public expr2t
{
public:
  expr2tc side_1, side_2;
  exists2t(const type2tc &type, const expr2tc &sym, const expr2tc &predicate)
    : expr2t(type, exists_id), side_1(sym), side_2(predicate)
  {
  }
  exists2t(const exists2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &exists2t::side_1, &exists2t::side_2);
  static std::string field_names[esbmct::num_type_fields];
};

class forall2t : public expr2t
{
public:
  expr2tc side_1, side_2;
  forall2t(const type2tc &type, const expr2tc &sym, const expr2tc &predicate)
    : expr2t(type, forall_id), side_1(sym), side_2(predicate)
  {
  }
  forall2t(const forall2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &forall2t::side_1, &forall2t::side_2);
  static std::string field_names[esbmct::num_type_fields];
};

/** Constant integer class.
 *  Records a constant integer of an arbitary precision, signed or unsigned.
 *  Simplification operations will cause the integer to be clipped to whatever
 *  bit size is in expr type.
 */
class constant_int2t : public expr2t
{
public:
  BigInt value;

  /** Primary constructor.
   *  @param type Type of this integer.
   *  @param input BigInt object containing the integer we're dealing with
   */
  constant_int2t(const type2tc &type, const BigInt &input)
    : expr2t(type, constant_int_id), value(input)
  {
  }
  constant_int2t(const constant_int2t &ref) = default;

  /** Accessor for fetching machine-word unsigned integer of this constant */
  unsigned long as_ulong() const;
  /** Accessor for fetching machine-word integer of this constant */
  long as_long() const;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &constant_int2t::value);
  static std::string field_names[esbmct::num_type_fields];
};

/** Constant fixedbv class. Records a fixed-width number in what I assume
 *  to be mantissa/exponent form, but which is described throughout CBMC code
 *  as fraction/integer parts. Stored in a fixedbvt.
 */
class constant_fixedbv2t : public expr2t
{
public:
  fixedbvt value;

  /** Primary constructor.
   *  @param type Type of this expression.
   *  @param value fixedbvt object containing number we'll be operating on
   */
  constant_fixedbv2t(const fixedbvt &value)
    : expr2t(value.spec.get_type(), constant_fixedbv_id), value(value)
  {
  }
  constant_fixedbv2t(const constant_fixedbv2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &constant_fixedbv2t::value);
  static std::string field_names[esbmct::num_type_fields];
};

/** Constant floatbv class. Records a floating-point number,
 *  Stored in a ieee_floatt.
 */
class constant_floatbv2t : public expr2t
{
public:
  ieee_floatt value;

  /** Primary constructor.
   *  @param type Type of this expression.
   *  @param value ieee_floatt object containing number we'll be operating on
   */
  constant_floatbv2t(const ieee_floatt &value)
    : expr2t(value.spec.get_type(), constant_floatbv_id), value(value)
  {
  }
  constant_floatbv2t(const constant_floatbv2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &constant_floatbv2t::value);
  static std::string field_names[esbmct::num_type_fields];
};

/** Constant boolean value.
 *  Contains a constant bool; rather self explanatory.
 */
class constant_bool2t : public expr2t
{
public:
  bool value;

  /** Primary constructor. @param value True or false */
  constant_bool2t(bool value)
    : expr2t(get_bool_type(), constant_bool_id), value(value)
  {
  }
  constant_bool2t(const constant_bool2t &ref) = default;

  /** Return whether contained boolean is true. */
  bool is_true() const;
  /** Return whether contained boolean is false. */
  bool is_false() const;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &constant_bool2t::value);
  static std::string field_names[esbmct::num_type_fields];
};

/** Constant class for string constants.
 *  Contains an irep_idt representing the constant string.
 */
class constant_string2t : public expr2t
{
public:
  using kindt = constant_string_kindt;

  irep_idt value;
  constant_string_kindt kind;

  /** Primary constructor.
   *  @param type Type of this string; presumably an array_type2t.
   *  @param stringref String pool'd string we're dealing with
   *  @param kind The kind of string literal:
   *              - DEFAULT: `""`
   *              - WIDE   : `L""`
   *              - UNICODE: `u8""`, `u""` and `U""`
   */
  constant_string2t(
    const type2tc &type,
    const irep_idt &stringref,
    constant_string_kindt kind)
    : expr2t(type, constant_string_id), value(stringref), kind(kind)
  {
  }
  constant_string2t(const constant_string2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &constant_string2t::value,
    &constant_string2t::kind);

  /** Convert string to a constant length array of characters */
  expr2tc to_array() const;

  /**
   * sizeof(literal)/sizeof(*literal), i.e., the number of elements in the
   * underlying array, including the '\0' terminator
   */
  size_t array_size() const;

  /**
   * Extract the i-th element from the string for i between 0 and
   * to_array_type(this->type).array_size (not the same as value.c_str()[i] when
   * to_array_type(this->type).subtype != char type)
   */
  expr2tc at(size_t i) const;

  static std::string field_names[esbmct::num_type_fields];
};

/** Constant structure.
 *  Contains a vector of expressions containing each member of the struct
 *  we're dealing with, corresponding to the types and field names in the
 *  struct_type2t type.
 */
class constant_struct2t : public expr2t
{
public:
  std::vector<expr2tc> datatype_members;

  /** Primary constructor.
   *  @param type Type of this structure, presumably a struct_type2t
   *  @param membrs Vector of member values that make up this struct.
   */
  constant_struct2t(const type2tc &type, const std::vector<expr2tc> &members)
    : expr2t(type, constant_struct_id), datatype_members(members)
  {
    // complex_type2t is a primitive subtype but lowers to a 2-element
    // struct view at the SMT boundary (see struct_union_members), and
    // the clang frontend builds complex literals as struct-id exprt
    // with a complex type — accept that shape here.
    assert(
      type->type_id == type2t::struct_id ||
      type->type_id == type2t::complex_id);
  }
  constant_struct2t(const constant_struct2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &constant_struct2t::datatype_members);
  static std::string field_names[esbmct::num_type_fields];
};

/** Constant union expression.
 *  Almost the same as constant_struct2t - a vector of members corresponding
 *  to the members described in the type. However, it seems the values pumped
 *  at us by CBMC only ever have one member (at position 0) representing the
 *  most recent value written to the union.
 */
class constant_union2t : public expr2t
{
public:
  std::vector<expr2tc> datatype_members;
  irep_idt init_field;

  /** Primary constructor.
   *  @param type Type of this structure, presumably a union_type2t
   *  @param members Vector of member values that make up this union.
   */
  constant_union2t(
    const type2tc &type,
    irep_idt if_,
    const std::vector<expr2tc> &members)
    : expr2t(type, constant_union_id),
      datatype_members(members),
      init_field(std::move(if_))
  {
    assert(is_union_type(type));
    // smt_conv.cpp's counterexample reconstruction intentionally builds unions
    //  with multiple members (see TODO in get_by_ast), so we can't check if the
    // union has at most 1 member initializer, with
    // assert(this->datatype_members.size() <= 1);
  }
  constant_union2t(const constant_union2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &constant_union2t::init_field,
    &constant_union2t::datatype_members);
  static std::string field_names[esbmct::num_type_fields];
};

/** Constant array.
 *  Contains a vector of array elements, pretty self explanatory. Only valid if
 *  its type has a constant sized array, can't have constant arrays of dynamic
 *  or infinitely sized arrays.
 */
class constant_array2t : public expr2t
{
public:
  std::vector<expr2tc> datatype_members;

  /** Primary constructor.
   *  @param type Type of this array, must be a constant sized array
   *  @param membrs Vector of elements in this array
   */
  constant_array2t(const type2tc &type, const std::vector<expr2tc> &members)
    : expr2t(type, constant_array_id), datatype_members(members)
  {
    assert(type->type_id == type2t::array_id);
  }
  constant_array2t(const constant_array2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &constant_array2t::datatype_members);
  static std::string field_names[esbmct::num_type_fields];
};

/** Constant array.
 *  Contains a vector of array elements, pretty self explanatory. Only valid if
 *  its type has a constant sized array, can't have constant arrays of dynamic
 *  or infinitely sized arrays.
 */
class constant_vector2t : public expr2t
{
public:
  std::vector<expr2tc> datatype_members;

  /** Primary constructor.
   *  @param type Type of this array, must be a constant sized array
   *  @param membrs Vector of elements in this array
   */
  constant_vector2t(const type2tc &type, const std::vector<expr2tc> &members)
    : expr2t(type, constant_vector_id), datatype_members(members)
  {
  }
  constant_vector2t(const constant_vector2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &constant_vector2t::datatype_members);
  static std::string field_names[esbmct::num_type_fields];
};

/** Constant array of one particular value.
 *  Expression with array type, possibly dynamic or infinitely sized, with
 *  all elements initialized to a single value.
 */
class constant_array_of2t : public expr2t
{
public:
  expr2tc initializer;

  /** Primary constructor.
   *  @param type Type of this expression, must be an array.
   *  @param init Initializer for each element in this array
   */
  constant_array_of2t(const type2tc &type, const expr2tc &init)
    : expr2t(type, constant_array_of_id), initializer(init)
  {
  }
  constant_array_of2t(const constant_array_of2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &constant_array_of2t::initializer);
  static std::string field_names[esbmct::num_type_fields];
};

/** Symbol type.
 *  Contains the name of some variable. Various levels of renaming.
 */
class symbol2t : public expr2t
{
public:
  using renaming_level = symbol_renaming_level;

  irep_idt thename;
  symbol_renaming_level rlevel;
  unsigned int level1_num;
  unsigned int level2_num;
  unsigned int thread_num;
  unsigned int node_num;

  /** Primary constructor
   *  @param type Type that this symbol has
   *  @param init Name of this symbol
   */
  symbol2t(
    const type2tc &type,
    const irep_idt &init,
    renaming_level lev = renaming_level::level0,
    unsigned int l1 = 0,
    unsigned int l2 = 0,
    unsigned int trd = 0,
    unsigned int node = 0)
    : expr2t(type, symbol_id),
      thename(init),
      rlevel(lev),
      level1_num(l1),
      level2_num(l2),
      thread_num(trd),
      node_num(node)
  {
    /* At some point in the past, symbols named "NULL" and "0" were equivalent.
     * The symbol called "0" should no longer be created for uniformity reasons.
     * Confirm that here, since support for it has been removed from smt_convt.
     * No other reason to disallow "0" as a symbol. */
    assert(init != "0");
  }

  symbol2t(const symbol2t &ref) = default;

  std::string get_symbol_name() const;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &symbol2t::thename,
    &symbol2t::rlevel,
    &symbol2t::level1_num,
    &symbol2t::level2_num,
    &symbol2t::thread_num,
    &symbol2t::node_num);
  static std::string field_names[esbmct::num_type_fields];
};

/** Nearbyint expression.
 *  Represents a rounding operation on a floatbv, we extend typecast as
 *  it already have a field for the rounding mode
 */
class nearbyint2t : public expr2t
{
public:
  expr2tc from;
  expr2tc rounding_mode;

  /** Primary constructor.
   *  @param type Type to round to
   *  @param from Expression to round from.
   *  @param rounding_mode Rounding mode, important only for floatbvs
   */
  nearbyint2t(
    const type2tc &type,
    const expr2tc &from_,
    const expr2tc &rounding_mode_)
    : expr2t(type, nearbyint_id), from(from_), rounding_mode(rounding_mode_)
  {
  }

  /** Primary constructor. This constructor defaults the rounding mode to
   *  the __ESBMC_rounding_mode symbol
   *  @param type Type to round to
   *  @param from Expression to round from.
   */
  nearbyint2t(const type2tc &type, const expr2tc &from_)
    : expr2t(type, nearbyint_id),
      from(from_),
      rounding_mode(symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode"))
  {
  }

  nearbyint2t(const nearbyint2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &nearbyint2t::from,
    &nearbyint2t::rounding_mode);
  static std::string field_names[esbmct::num_type_fields];
};

/** Typecast expression.
 *  Represents cast from contained expression 'from' to the type of this
 *  typecast.
 */
class typecast2t : public expr2t
{
public:
  expr2tc from;
  expr2tc rounding_mode;

  /** Primary constructor.
   *  @param type Type to typecast to
   *  @param from Expression to cast from.
   *  @param rounding_mode Rounding mode, important only for floatbvs
   */
  typecast2t(
    const type2tc &type,
    const expr2tc &from_,
    const expr2tc &rounding_mode_)
    : expr2t(type, typecast_id), from(from_), rounding_mode(rounding_mode_)
  {
  }

  /** Primary constructor. This constructor defaults the rounding mode to
   *  the __ESBMC_rounding_mode symbol
   *  @param type Type to typecast to
   *  @param from Expression to cast from.
   */
  typecast2t(const type2tc &type, const expr2tc &from_)
    : expr2t(type, typecast_id),
      from(from_),
      rounding_mode(symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode"))
  {
  }

  typecast2t(const typecast2t &ref) = default;
  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &typecast2t::from,
    &typecast2t::rounding_mode);
  static std::string field_names[esbmct::num_type_fields];
};

/** Bitcast expression.
 *  Represents cast from contained expression 'from' to the type of this
 *  typecast... but where the cast is performed at a 'bit representation' level.
 *  That is: the 'from' field is not interpreted by its logical value, but
 *  instead by the corresponding bit representation. The prime example of this
 *  is bitcasting floats: if one typecasted them to integers, they would be
 *  rounded; bitcasting them produces the bit-representation of the float, as
 *  an integer value.
 *
 *  Bitcasts are only allowed between types of equal width.
 *
 */
class bitcast2t : public expr2t
{
public:
  expr2tc from;

  /** Primary constructor.
   *  @param type Type to bitcast to
   *  @param from Expression to cast from.
   */
  bitcast2t(const type2tc &type, const expr2tc &from_)
    : expr2t(type, bitcast_id), from(from_)
  {
    try
    {
      assert(type->get_width() == from_->type->get_width());
    }
    catch (const type2t::symbolic_type_excp &)
    {
      /* ignore */
    }
  }

  bitcast2t(const bitcast2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &bitcast2t::from);
  static std::string field_names[esbmct::num_type_fields];
};

/** If-then-else expression.
 *  Represents a ternary operation, (cond) ? truevalue : falsevalue.
 */
class if2t : public expr2t
{
public:
  expr2tc cond;
  expr2tc true_value;
  expr2tc false_value;

  /** Primary constructor
   *  @param type Type this expression evaluates to.
   *  @param cond Condition to evaulate which side of ternary operator is used.
   *  @param trueval Value to use if cond evaluates to true.
   *  @param falseval Value to use if cond evaluates to false.
   */
  if2t(
    const type2tc &type,
    const expr2tc &cond_,
    const expr2tc &trueval,
    const expr2tc &falseval)
    : expr2t(type, if_id),
      cond(cond_),
      true_value(trueval),
      false_value(falseval)
  {
    assert(type->type_id == trueval->type->type_id);
    assert(type->type_id == falseval->type->type_id);
  }
  if2t(const if2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &if2t::cond,
    &if2t::true_value,
    &if2t::false_value);
  static std::string field_names[esbmct::num_type_fields];
};

/** Defines one of the six binary boolean relation nodes. Each takes two
 *  operands of any matching scalar/pointer type, has boolean result type,
 *  and provides an out-of-line do_simplify override in
 *  src/util/expr_simplifier.cpp. */
#define ESBMC_DEFINE_RELATION2T(name)                                          \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc side_1, side_2;                                                    \
    name##2t(const expr2tc &v1, const expr2tc &v2)                             \
      : expr2t(get_bool_type(), name##_id), side_1(v1), side_2(v2)             \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields =                                             \
      std::make_tuple(&expr2t::type, &name##2t ::side_1, &name##2t ::side_2);  \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_RELATION2T(equality);
ESBMC_DEFINE_RELATION2T(notequal);
ESBMC_DEFINE_RELATION2T(lessthan);
ESBMC_DEFINE_RELATION2T(greaterthan);
ESBMC_DEFINE_RELATION2T(lessthanequal);
ESBMC_DEFINE_RELATION2T(greaterthanequal);
#undef ESBMC_DEFINE_RELATION2T

/* The macros below fold sets of `*2t` classes that share a constructor
 * shape and behaviour. Each generated class still inherits directly
 * from expr2t and owns its own fields; the macro just spares the
 * repetition of writing six near-identical class bodies. */

/** Arithmetic two-operand node (`add`/`sub`/`mul`/`div`/`modulus`). */
#define ESBMC_DEFINE_ARITH_2OP(name)                                           \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc side_1, side_2;                                                    \
    name##2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)        \
      : expr2t(type, name##_id), side_1(v1), side_2(v2)                        \
    {                                                                          \
      assert_arith_2ops_consistency(type, name##_id, v1, v2);                  \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields =                                             \
      std::make_tuple(&expr2t::type, &name##2t ::side_1, &name##2t ::side_2);  \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_ARITH_2OP(add);
ESBMC_DEFINE_ARITH_2OP(sub);
ESBMC_DEFINE_ARITH_2OP(mul);
ESBMC_DEFINE_ARITH_2OP(div);
ESBMC_DEFINE_ARITH_2OP(modulus);
#undef ESBMC_DEFINE_ARITH_2OP

/** Bitwise / shift two-operand node (`bitand`/`bitor`/`bitxor`/
 *  `shl`/`ashr`/`lshr`). */
#define ESBMC_DEFINE_BIT_2OP(name)                                             \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc side_1, side_2;                                                    \
    name##2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)        \
      : expr2t(type, name##_id), side_1(v1), side_2(v2)                        \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields =                                             \
      std::make_tuple(&expr2t::type, &name##2t ::side_1, &name##2t ::side_2);  \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_BIT_2OP(bitand);
ESBMC_DEFINE_BIT_2OP(bitor);
ESBMC_DEFINE_BIT_2OP(bitxor);
ESBMC_DEFINE_BIT_2OP(lshr);
ESBMC_DEFINE_BIT_2OP(shl);
ESBMC_DEFINE_BIT_2OP(ashr);
#undef ESBMC_DEFINE_BIT_2OP

/** Arithmetic one-operand node (`neg`/`abs`/`bswap`/`bitnot`). */
#define ESBMC_DEFINE_ARITH_1OP(name)                                           \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc value;                                                             \
    name##2t(const type2tc &type, const expr2tc &v)                            \
      : expr2t(type, name##_id), value(v)                                      \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields =                                             \
      std::make_tuple(&expr2t::type, &name##2t ::value);                       \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_ARITH_1OP(neg);
ESBMC_DEFINE_ARITH_1OP(abs);
ESBMC_DEFINE_ARITH_1OP(bitnot);
ESBMC_DEFINE_ARITH_1OP(bswap);
#undef ESBMC_DEFINE_ARITH_1OP

/** Pointer one-operand node (`pointer_object`/`pointer_capability`). */
#define ESBMC_DEFINE_POINTER_1OP(name)                                         \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc ptr_obj;                                                           \
    name##2t(const type2tc &type, const expr2tc &v)                            \
      : expr2t(type, name##_id), ptr_obj(v)                                    \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields =                                             \
      std::make_tuple(&expr2t::type, &name##2t ::ptr_obj);                     \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_POINTER_1OP(pointer_object);
ESBMC_DEFINE_POINTER_1OP(pointer_capability);
#undef ESBMC_DEFINE_POINTER_1OP

/** Logical two-operand boolean-result node. Used for `and`/`or`/`xor`/
 *  `implies` and the Python runtime predicates `isinstance`/`hasattr`/
 *  `isnone`. Implicit `get_bool_type()` result. */
#define ESBMC_DEFINE_LOGIC_2OP(name)                                           \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc side_1, side_2;                                                    \
    name##2t(const expr2tc &s1, const expr2tc &s2)                             \
      : expr2t(get_bool_type(), name##_id), side_1(s1), side_2(s2)             \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields =                                             \
      std::make_tuple(&expr2t::type, &name##2t ::side_1, &name##2t ::side_2);  \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_LOGIC_2OP(and);
ESBMC_DEFINE_LOGIC_2OP(or);
ESBMC_DEFINE_LOGIC_2OP(xor);
ESBMC_DEFINE_LOGIC_2OP(implies);
ESBMC_DEFINE_LOGIC_2OP(isinstance);
ESBMC_DEFINE_LOGIC_2OP(hasattr);
ESBMC_DEFINE_LOGIC_2OP(isnone);
#undef ESBMC_DEFINE_LOGIC_2OP

/** FP classification single-operand predicate (`isnan`/`isinf`/
 *  `isnormal`/`isfinite`). Implicit `get_bool_type()` result. */
#define ESBMC_DEFINE_FP_PREDICATE_1OP(name)                                    \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc value;                                                             \
    name##2t(const expr2tc &operand)                                           \
      : expr2t(get_bool_type(), name##_id), value(operand)                     \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields = std::make_tuple(&name##2t ::value);         \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_FP_PREDICATE_1OP(isnan);
ESBMC_DEFINE_FP_PREDICATE_1OP(isinf);
ESBMC_DEFINE_FP_PREDICATE_1OP(isnormal);
ESBMC_DEFINE_FP_PREDICATE_1OP(isfinite);
#undef ESBMC_DEFINE_FP_PREDICATE_1OP

/** Pointer-object boolean predicate (`valid_object`/`races_check`/
 *  `deallocated_obj`). Implicit `get_bool_type()` result. */
#define ESBMC_DEFINE_OBJECT_PREDICATE_1OP(name)                                \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc value;                                                             \
    name##2t(const expr2tc &operand)                                           \
      : expr2t(get_bool_type(), name##_id), value(operand)                     \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields = std::make_tuple(&name##2t ::value);         \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_OBJECT_PREDICATE_1OP(valid_object);
ESBMC_DEFINE_OBJECT_PREDICATE_1OP(races_check);
ESBMC_DEFINE_OBJECT_PREDICATE_1OP(deallocated_obj);
#undef ESBMC_DEFINE_OBJECT_PREDICATE_1OP

/** Pointer-object size-returning op (`capability_base`/`capability_top`/
 *  `dynamic_size`). Implicit `size_type2()` result. */
#define ESBMC_DEFINE_OBJECT_SIZE_1OP(name)                                     \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc value;                                                             \
    name##2t(const expr2tc &operand)                                           \
      : expr2t(size_type2(), name##_id), value(operand)                        \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields = std::make_tuple(&name##2t ::value);         \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_OBJECT_SIZE_1OP(dynamic_size);
ESBMC_DEFINE_OBJECT_SIZE_1OP(capability_base);
ESBMC_DEFINE_OBJECT_SIZE_1OP(capability_top);
#undef ESBMC_DEFINE_OBJECT_SIZE_1OP

/** Single-operand overflow-family op returning int32 (`signbit`/
 *  `popcount`). */
#define ESBMC_DEFINE_OVERFLOW_INT32_1OP(name)                                  \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc operand;                                                           \
    name##2t(const expr2tc &op)                                                \
      : expr2t(get_int32_type(), name##_id), operand(op)                       \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields =                                             \
      std::make_tuple(&expr2t::type, &name##2t ::operand);                     \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_OVERFLOW_INT32_1OP(signbit);
ESBMC_DEFINE_OVERFLOW_INT32_1OP(popcount);
#undef ESBMC_DEFINE_OVERFLOW_INT32_1OP

/** Marker node holding only a `type` (no operands). Used for
 *  `unknown`/`invalid`/`null_object`. */
#define ESBMC_DEFINE_TYPE_ONLY(name)                                           \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    name##2t(const type2tc &type) : expr2t(type, name##_id)                    \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields = std::make_tuple(&expr2t::type);             \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_TYPE_ONLY(unknown);
ESBMC_DEFINE_TYPE_ONLY(invalid);
ESBMC_DEFINE_TYPE_ONLY(null_object);
#undef ESBMC_DEFINE_TYPE_ONLY

/** `code_*` statement with empty type and a single `expr2tc` operand
 *  (`code_expression`/`code_return`/`code_free`/`code_cpp_del_array`/
 *  `code_cpp_delete`). */
#define ESBMC_DEFINE_CODE_EXPRESSION_1OP(name)                                 \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc operand;                                                           \
    locationt location; /* not reflected: source loc travels with the stmt */  \
    static constexpr std::size_t excluded_field_bytes = sizeof(locationt);     \
    name##2t(const expr2tc &op, const locationt &loc = locationt())            \
      : expr2t(get_empty_type(), name##_id), operand(op), location(loc)        \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields = std::make_tuple(&name##2t ::operand);       \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_CODE_EXPRESSION_1OP(code_expression);
ESBMC_DEFINE_CODE_EXPRESSION_1OP(code_return);
ESBMC_DEFINE_CODE_EXPRESSION_1OP(code_free);
ESBMC_DEFINE_CODE_EXPRESSION_1OP(code_cpp_del_array);
ESBMC_DEFINE_CODE_EXPRESSION_1OP(code_cpp_delete);
#undef ESBMC_DEFINE_CODE_EXPRESSION_1OP

/** `code_*` declaration carrying `(type, irep_idt name)`. Used for
 *  `code_decl`/`code_dead`. */
#define ESBMC_DEFINE_CODE_DECL(name)                                           \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    irep_idt value;                                                            \
    locationt location; /* not reflected: source loc travels with the stmt */  \
    static constexpr std::size_t excluded_field_bytes = sizeof(locationt);     \
    name##2t(                                                                  \
      const type2tc &type,                                                     \
      const irep_idt &n,                                                       \
      const locationt &loc = locationt())                                      \
      : expr2t(type, name##_id), value(n), location(loc)                       \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields =                                             \
      std::make_tuple(&expr2t::type, &name##2t ::value);                       \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_CODE_DECL(code_dead);
#undef ESBMC_DEFINE_CODE_DECL

/** `code_decl2t` — variable declaration, with optional initializer.
 *
 *  The `init` field carries the initializer expression when the source form
 *  is a 2-operand `code_decl(symbol, init)`.  It is nil when there is no
 *  initializer (1-operand form).  Keeping the initializer here (rather than
 *  splitting into a separate `code_block`) ensures that `goto_convert` places
 *  the DEAD instruction at the correct scope boundary instead of immediately
 *  after the assignment. */
class code_decl2t : public expr2t
{
public:
  irep_idt value; // symbol name
  expr2tc init;   // optional initializer; nil when absent
  locationt location; // not reflected: source loc travels with the stmt
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);
  code_decl2t(
    const type2tc &type,
    const irep_idt &n,
    const expr2tc &i = expr2tc(),
    const locationt &loc = locationt())
    : expr2t(type, code_decl_id), value(n), init(i), location(loc)
  {
  }
  code_decl2t(const code_decl2t &ref) = default;
  expr2tc do_simplify() const override;
  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &code_decl2t::value, &code_decl2t::init);
  static std::string field_names[esbmct::num_type_fields];
};

/** C++20 three-way comparison `a <=> b`. Result type is the
 * comparison-category struct (`std::strong_ordering` /
 * `std::weak_ordering` / `std::partial_ordering`); the discriminating
 * signed-char member sits at the start of the struct. The expansion to
 *
 *   side_1 <  side_2  ->  T{-1}    (less)
 *   side_1 == side_2  ->  T{ 0}    (equivalent / equal)
 *   else              ->  T{ 1}    (greater)
 *
 * is performed at the SMT layer rather than the AST level so the
 * semantic node survives through symex / value_set / interval analysis,
 * and operands are captured once.  Per [expr.spaceship] in N4861.
 **/
class cmp_three_way2t : public expr2t
{
public:
  expr2tc side_1;
  expr2tc side_2;

  cmp_three_way2t(const type2tc &t, const expr2tc &v1, const expr2tc &v2)
    : expr2t(t, cmp_three_way_id), side_1(v1), side_2(v2)
  {
  }
  cmp_three_way2t(const cmp_three_way2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &cmp_three_way2t::side_1,
    &cmp_three_way2t::side_2);
  static std::string field_names[esbmct::num_type_fields];
};

/** Not operation. Inverts boolean operand. Always has boolean type. */
class not2t : public expr2t
{
public:
  expr2tc value;

  /** Primary constructor. @param val Boolean typed operand to invert. */
  not2t(const expr2tc &val) : expr2t(get_bool_type(), not_id), value(val)
  {
  }
  not2t(const not2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(&not2t::value);
  static std::string field_names[esbmct::num_type_fields];
};

/** Defines an IEEE two-operand floating-point arithmetic node
 *  (add/sub/mul/div). Each takes two operands and a rounding mode,
 *  has matching floatbv operand/result types, and provides an
 *  out-of-line do_simplify override in src/util/expr_simplifier.cpp. */
#define ESBMC_DEFINE_IEEE_ARITH_2OP(name)                                      \
  class name##2t : public expr2t                                               \
  {                                                                            \
  public:                                                                      \
    expr2tc rounding_mode, side_1, side_2;                                     \
    name##2t(                                                                  \
      const type2tc &type,                                                     \
      const expr2tc &v1,                                                       \
      const expr2tc &v2,                                                       \
      const expr2tc &rm)                                                       \
      : expr2t(type, name##_id), rounding_mode(rm), side_1(v1), side_2(v2)     \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static constexpr auto fields = std::make_tuple(                            \
      &expr2t::type,                                                           \
      &name##2t ::rounding_mode,                                               \
      &name##2t ::side_1,                                                      \
      &name##2t ::side_2);                                                     \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_IEEE_ARITH_2OP(ieee_add);
ESBMC_DEFINE_IEEE_ARITH_2OP(ieee_sub);
ESBMC_DEFINE_IEEE_ARITH_2OP(ieee_mul);
ESBMC_DEFINE_IEEE_ARITH_2OP(ieee_div);
#undef ESBMC_DEFINE_IEEE_ARITH_2OP

/** IEEE fused multiply-add operation. Computes (x*y) + z as if to infinite
 *  precision and rounded only once to fit the result type. Must be
 *  floatbvs types. Types of the 3 operands and expr type should match. */
class ieee_fma2t : public expr2t
{
public:
  expr2tc rounding_mode;
  expr2tc value_1;
  expr2tc value_2;
  expr2tc value_3;

  /** Primary constructor.
   *  @param type Type of this expr.
   *  @param v1 First operand.
   *  @param v2 Second operand.
   *  @param v3 Second operand.
   *  @param rm rounding mode. */
  ieee_fma2t(
    const type2tc &type,
    const expr2tc &v1,
    const expr2tc &v2,
    const expr2tc &v3,
    const expr2tc &rm)
    : expr2t(type, ieee_fma_id),
      rounding_mode(rm),
      value_1(v1),
      value_2(v2),
      value_3(v3)
  {
  }
  ieee_fma2t(const ieee_fma2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &ieee_fma2t::value_1,
    &ieee_fma2t::value_2,
    &ieee_fma2t::value_3,
    &ieee_fma2t::rounding_mode);
  static std::string field_names[esbmct::num_type_fields];
};

/** IEEE sqrt operation. Square root of the first operand. Must be a
 *  floatbv. */
class ieee_sqrt2t : public expr2t
{
public:
  expr2tc rounding_mode;
  expr2tc value;

  /** Primary constructor.
   *  @param type Type of this expr.
   *  @param v1 Operand to take the square root of.
   *  @param rm Rounding mode. */
  ieee_sqrt2t(const type2tc &type, const expr2tc &v1, const expr2tc &rm)
    : expr2t(type, ieee_sqrt_id), rounding_mode(rm), value(v1)
  {
  }
  ieee_sqrt2t(const ieee_sqrt2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &ieee_sqrt2t::value,
    &ieee_sqrt2t::rounding_mode);
  static std::string field_names[esbmct::num_type_fields];
};

/** Same-object operation. Checks whether two operands with pointer type have the
 *  same pointer object or not. Always has boolean result.
 * */
class same_object2t : public expr2t
{
public:
  expr2tc side_1;
  expr2tc side_2;

  /** Primary constructor. @param v1 First object. @param v2 Second object. */
  same_object2t(const expr2tc &v1, const expr2tc &v2)
    : expr2t(get_bool_type(), same_object_id), side_1(v1), side_2(v2)
  {
  }
  same_object2t(const same_object2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &same_object2t::side_1,
    &same_object2t::side_2);
  static std::string field_names[esbmct::num_type_fields];
};

/** Extract pointer offset. From an expression of pointer type, produce the
 *  number of bytes difference between where this pointer points to and the start
 *  of the object it points at. */
class pointer_offset2t : public expr2t
{
public:
  expr2tc ptr_obj;

  /** Primary constructor.
   *  @param type Model basic integer type.
   *  @param ptrobj Pointer object to get offset from. */
  pointer_offset2t(const type2tc &type, const expr2tc &ptrobj)
    : expr2t(type, pointer_offset_id), ptr_obj(ptrobj)
  {
    assert(type->type_id == type2t::signedbv_id);
    assert(type->get_width() == config.ansi_c.address_width);
  }
  pointer_offset2t(const pointer_offset2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &pointer_offset2t::ptr_obj);
  static std::string field_names[esbmct::num_type_fields];
};

/** Address of operation. Takes some object as an argument - ideally a symbol
 *  renamed to level 1, unfortunately some string constants reach here. Produces
 *  pointer typed expression. */
class address_of2t : public expr2t
{
public:
  expr2tc ptr_obj;

  /** Primary constructor.
   *  @param subtype Subtype of pointer to generate. Crucially, the type of the
   *         expr is a pointer to this subtype. This is slightly unintuitive,
   *         might be changed in the future.
   *  @param ptrobj Item to take pointer to. */
  address_of2t(const type2tc &subtype, const expr2tc &ptrobj)
    : expr2t(pointer_type2tc(subtype), address_of_id), ptr_obj(ptrobj)
  {
    assert(ptrobj->expr_id != expr2t::constant_int_id);
    assert(ptrobj->expr_id != expr2t::address_of_id);
  }
  address_of2t(const address_of2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &address_of2t::ptr_obj);
  static std::string field_names[esbmct::num_type_fields];
};

/** Extract byte from data. From a particular data structure, extracts a single
 *  byte from its byte representation, at a particular offset into the data
 *  structure. Must only evaluate to byte types.
 * */
class byte_extract2t : public expr2t
{
public:
  expr2tc source_value;
  expr2tc source_offset;
  bool big_endian;

  /** Primary constructor.
   *  @param type Type of this expression. May only ever be an 8 bit integer
   *  @param is_big_endian Whether or not to use big endian byte representation
   *         of source object.
   *  @param source Object to extract data from. Any type.
   *  @param offset Offset into source data object to extract from. */
  byte_extract2t(
    const type2tc &type,
    const expr2tc &source,
    const expr2tc &offset,
    bool is_big_endian)
    : expr2t(type, byte_extract_id),
      source_value(source),
      source_offset(offset),
      big_endian(is_big_endian)
  {
  }
  byte_extract2t(const byte_extract2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &byte_extract2t::source_value,
    &byte_extract2t::source_offset,
    &byte_extract2t::big_endian);
  static std::string field_names[esbmct::num_type_fields];
};

/** Update byte. Takes a data object and updates the value of a particular
 *  byte in its byte representation, at a particular offset into the data object.
 *  Output of expression is a new copy of the source object, with the updated
 *  value. */
class byte_update2t : public expr2t
{
public:
  expr2tc source_value;
  expr2tc source_offset;
  expr2tc update_value;
  bool big_endian;

  /** Primary constructor
   *  @param type Type of resulting, updated, data object.
   *  @param is_big_endian Whether to use big endian byte representation.
   *  @param source Source object in which to update a byte.
   *  @param updateval Value of byte to  update source with. */
  byte_update2t(
    const type2tc &type,
    const expr2tc &source,
    const expr2tc &offset,
    const expr2tc &updateval,
    bool is_big_endian)
    : expr2t(type, byte_update_id),
      source_value(source),
      source_offset(offset),
      update_value(updateval),
      big_endian(is_big_endian)
  {
  }
  byte_update2t(const byte_update2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &byte_update2t::source_value,
    &byte_update2t::source_offset,
    &byte_update2t::update_value,
    &byte_update2t::big_endian);
  static std::string field_names[esbmct::num_type_fields];
};

/** With operation. Updates either an array or a struct/union with a new element
 *  or member. Expression value is the array or struct/union with the updated
 *  value. Ideally in the future this will become two operations, one for arrays
 *  and one for structs/unions. */
class with2t : public expr2t
{
  void assert_consistency() const;

public:
  expr2tc source_value;
  expr2tc update_field;
  expr2tc update_value;

  /** Primary constructor.
   *  @param type Type of this expression; Same as source.
   *  @param source Data object to update.
   *  @param field Field to update - a constant string naming the field if source
   *         is a struct/union, or an integer index if source is an array. */
  with2t(
    const type2tc &type,
    const expr2tc &source,
    const expr2tc &field,
    const expr2tc &value)
    : expr2t(type, with_id),
      source_value(source),
      update_field(field),
      update_value(value)
  {
#ifndef NDEBUG
    assert_consistency();
#endif
  }
  with2t(const with2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &with2t::source_value,
    &with2t::update_field,
    &with2t::update_value);
  static std::string field_names[esbmct::num_type_fields];
};

/** Member operation. Extracts a particular member out of a struct or union.
 * */
class member2t : public expr2t
{
public:
  expr2tc source_value;
  irep_idt member;

  /** Primary constructor.
   *  @param type Type of extracted member.
   *  @param source Data structure to extract from.
   *  @param memb Name of member to extract.  */
  member2t(const type2tc &type, const expr2tc &source, const irep_idt &memb)
    : expr2t(type, member_id), source_value(source), member(memb)
  {
#ifndef NDEBUG /* only check consistency in non-Release builds */
    /* The source is normally a resolved struct/union/complex. A `symbol_id`
       source is permitted ONLY as a transient pre-resolution state: the
       IREP2-migration V.1k "two-phase source invariant" lets a frontend build a
       member2t before type resolution (the Python converter, ahead of the
       IREP2-native adjuster) hand a by-name `symbol_type2t` here; the adjuster
       MUST follow it to a struct before symex. The strong invariant is
       re-enforced post-adjust by that pass, not dropped. No existing frontend
       builds member2t pre-adjust (all go through migrate at goto-convert, which
       is post-adjust), so this disjunct is staged enabling infra, exercised
       once the V.1k converter/adjuster pilot lands. */
    assert(
      source->type->type_id == type2t::struct_id ||
      source->type->type_id == type2t::union_id ||
      source->type->type_id == type2t::complex_id ||
      source->type->type_id == type2t::symbol_id);
    /* member must exist exactly once in the parent struct/union — only checkable
       once the source type is resolved (skipped for the transient symbol case) */
    assert(
      source->type->type_id == type2t::symbol_id ||
      struct_union_get_component_number(source->type, memb).has_value());
#endif
  }
  member2t(const member2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &member2t::source_value, &member2t::member);
  static std::string field_names[esbmct::num_type_fields];
};

/** Member reference
 * */
class member_ref2t : public expr2t
{
public:
  irep_idt member;

  /** Primary constructor.
   *  @param type Type of extracted member.
   *  @param memb Name of member to extract.  */
  member_ref2t(const type2tc &type, const irep_idt &memb)
    : expr2t(type, member_ref_id), member(memb)
  {
  }
  member_ref2t(const member_ref2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &member_ref2t::member);
  static std::string field_names[esbmct::num_type_fields];
};

/** Member pointer
 * */
class ptr_mem2t : public expr2t
{
public:
  expr2tc source_value;
  expr2tc member_pointer;

  /** Primary constructor.
   *  @param source Data structure to extract from.
   *  @param pointer Pointer to member.  */
  ptr_mem2t(const type2tc &type, const expr2tc &source, const expr2tc &pointer)
    : expr2t(type, ptr_mem_id), source_value(source), member_pointer(pointer)
  {
  }
  ptr_mem2t(const ptr_mem2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &ptr_mem2t::source_value,
    &ptr_mem2t::member_pointer);
  static std::string field_names[esbmct::num_type_fields];
};

/** Array index operation. Extracts an element from an array at a particular
 *  index. */
class index2t : public expr2t
{
public:
  expr2tc source_value;
  expr2tc index;

  /** Primary constructor.
   *  @param type Type of element extracted.
   *  @param source Array to extract data from.
   *  @param index Element in source to extract from. */
  index2t(const type2tc &type, const expr2tc &source, const expr2tc &idx)
    : expr2t(type, index_id), source_value(source), index(idx)
  {
    /* A `symbol_id` source is permitted only as a transient pre-resolution
       state (V.1k two-phase source invariant, see member2t above); the
       IREP2-native adjuster resolves it to an array/vector before symex. */
    assert(
      is_array_type(source) || is_vector_type(source) ||
      source->type->type_id == type2t::symbol_id);
  }
  index2t(const index2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &index2t::source_value, &index2t::index);
  static std::string field_names[esbmct::num_type_fields];
};

/** Check whether operand overflows. Operand must be either add, subtract,
 *  or multiply, and have integer operands themselves. If the result of the
 *  operation doesn't fit in the bitwidth of the operands, this expr evaluates
 *  to true. XXXjmorse - in the future we should ensure the type of the
 *  operand is the expected type result of the operation. That way we can tell
 *  whether to do a signed or unsigned over/underflow test. */
class overflow2t : public expr2t
{
public:
  expr2tc operand;

  /** Primary constructor.
   *  @param operand Operation to test overflow on; either an add, subtract, or
   *         multiply. */
  overflow2t(const expr2tc &operand_)
    : expr2t(get_bool_type(), overflow_id), operand(operand_)
  {
  }
  overflow2t(const overflow2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &overflow2t::operand);
  static std::string field_names[esbmct::num_type_fields];
};

/** Test if a cast overflows. Check to see whether casting the operand to a
 *  particular bitsize will cause an integer overflow. If it does, this expr
 *  evaluates to true. */
class overflow_cast2t : public expr2t
{
public:
  expr2tc operand;
  unsigned int bits;

  /** Primary constructor.
   *  @param operand Value to test cast out on. Should have integer type.
   *  @param bits Number of integer bits to cast operand to.  */
  overflow_cast2t(const expr2tc &operand_, unsigned int bits_)
    : expr2t(get_bool_type(), overflow_cast_id), operand(operand_), bits(bits_)
  {
  }
  overflow_cast2t(const overflow_cast2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &overflow_cast2t::operand,
    &overflow_cast2t::bits);
  static std::string field_names[esbmct::num_type_fields];
};

/** Test for negation overflows. Check whether or not negating an operand would
 *  lead to an integer overflow - for example, there's no representation of
 *  -INT_MIN. Evaluates to true if overflow would occur. */
class overflow_neg2t : public expr2t
{
public:
  expr2tc operand;

  /** Primary constructor. @param operand Integer to test negation of. */
  overflow_neg2t(const expr2tc &operand_)
    : expr2t(get_bool_type(), overflow_neg_id), operand(operand_)
  {
  }
  overflow_neg2t(const overflow_neg2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &overflow_neg2t::operand);
  static std::string field_names[esbmct::num_type_fields];
};

/** Record a dynamicly allocated object. Exclusively for use in pointer analysis.
 * */
class dynamic_object2t : public expr2t
{
public:
  expr2tc instance;
  bool invalid;
  bool unknown;

  dynamic_object2t(
    const type2tc &type,
    const expr2tc &inst,
    bool inv,
    bool uknown)
    : expr2t(type, dynamic_object_id),
      instance(inst),
      invalid(inv),
      unknown(uknown)
  {
  }
  dynamic_object2t(const dynamic_object2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &dynamic_object2t::instance,
    &dynamic_object2t::invalid,
    &dynamic_object2t::unknown);
  static std::string field_names[esbmct::num_type_fields];
};

/** Dereference operation. Expanded by symbolic execution into an if-then-else
 *  set of cases that take the value set of what this pointer might point at,
 *  examines the pointer's pointer object, and constructs a huge if-then-else
 *  case to evaluate to the appropriate data object for this pointer.
 * */
class dereference2t : public expr2t
{
public:
  expr2tc value;

  /** Primary constructor.
   *  @param type Type of dereferenced data.
   *  @param operand Pointer to dereference. */
  dereference2t(const type2tc &type, const expr2tc &operand)
    : expr2t(type, dereference_id), value(operand)
  {
  }
  dereference2t(const dereference2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &dereference2t::value);
  static std::string field_names[esbmct::num_type_fields];
};

/** Irep for various side effects. Stores data about various things that can
 *  cause side effects, such as memory allocations, nondeterministic value
 *  allocations (nondet_* funcs,).
 *
 *  Also allows for function-calls to be represented. This side-effect
 *  expression is how function calls inside expressions are represented during
 *  parsing, and are all flattened out prior to GOTO program creation. However,
 *  under certain circumstances irep2 needs to represent such function calls,
 *  so this facility is preserved in irep2.
 *
 * */
class sideeffect2t : public expr2t
{
public:
  using allockind = sideeffect_allockind;

  expr2tc operand;
  expr2tc size;
  std::vector<expr2tc> arguments;
  type2tc alloctype;
  sideeffect_allockind kind;

  /** Primary constructor.
   *  @param t Type this side-effect evaluates to.
   *  @param operand Not really certain. Sometimes turns up in string-irep.
   *  @param sz Size of dynamic allocation to make.
   *  @param alloct Type of piece of data to allocate.
   *  @param a Vector of arguments to function call. */
  sideeffect2t(
    const type2tc &t,
    const expr2tc &oper,
    const expr2tc &sz,
    const std::vector<expr2tc> &a,
    const type2tc &alloct,
    sideeffect_allockind k)
    : expr2t(t, sideeffect_id),
      operand(oper),
      size(sz),
      arguments(a),
      alloctype(alloct),
      kind(k)
  {
    if (k == sideeffect_allockind::alloca)
      assert(oper->type == sz->type);
  }
  sideeffect2t(const sideeffect2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &sideeffect2t::operand,
    &sideeffect2t::size,
    &sideeffect2t::arguments,
    &sideeffect2t::alloctype,
    &sideeffect2t::kind);
  static std::string field_names[esbmct::num_type_fields];
};

class code_block2t : public expr2t
{
public:
  std::vector<expr2tc> operands;
  locationt location; // not reflected: source loc travels with the stmt
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_block2t(
    const std::vector<expr2tc> &ops,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_block_id), operands(ops), location(loc)
  {
  }
  code_block2t(const code_block2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &code_block2t::operands);
  static std::string field_names[esbmct::num_type_fields];
};

class code_assign2t : public expr2t
{
public:
  expr2tc target;
  expr2tc source;
  locationt location; // not reflected: source loc travels with the stmt
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_assign2t(
    const expr2tc &tgt,
    const expr2tc &src,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_assign_id),
      target(tgt),
      source(src),
      location(loc)
  {
  }
  code_assign2t(const code_assign2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &code_assign2t::target,
    &code_assign2t::source);
  static std::string field_names[esbmct::num_type_fields];
};

class code_printf2t : public expr2t
{
public:
  std::vector<expr2tc> operands;
  printf_kindt kind;
  locationt location; // not reflected: source loc travels with the stmt
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_printf2t(
    const std::vector<expr2tc> &opers,
    printf_kindt k,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_printf_id),
      operands(opers),
      kind(k),
      location(loc)
  {
  }
  code_printf2t(const code_printf2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &code_printf2t::operands,
    &code_printf2t::kind);
  static std::string field_names[esbmct::num_type_fields];
};

class code_skip2t : public expr2t
{
public:
  locationt location; // not reflected: source loc travels with the stmt
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);
  code_skip2t(const type2tc &type, const locationt &loc = locationt())
    : expr2t(type, code_skip_id), location(loc)
  {
  }
  code_skip2t(const code_skip2t &ref) = default;

  static constexpr auto fields = std::make_tuple(&expr2t::type);
  static std::string field_names[esbmct::num_type_fields];
};

/** C++ constructor "this" placeholder (`exprt("new_object")`). Appears inside
 *  a `temporary_object` initializer before `replace_new_object` substitutes
 *  the real symbol. Carries only the struct type being constructed. */
class new_object2t : public expr2t
{
public:
  new_object2t(const type2tc &type) : expr2t(type, new_object_id)
  {
  }
  new_object2t(const new_object2t &ref) = default;

  expr2tc do_simplify() const override;
  static constexpr auto fields = std::make_tuple(&expr2t::type);
  static std::string field_names[esbmct::num_type_fields];
};

class code_goto2t : public expr2t
{
public:
  irep_idt target;
  locationt location; // not reflected: source loc travels with the stmt
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_goto2t(const irep_idt &targ, const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_goto_id), target(targ), location(loc)
  {
  }
  code_goto2t(const code_goto2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &code_goto2t::target);
  static std::string field_names[esbmct::num_type_fields];
};

class object_descriptor2t : public expr2t
{
public:
  expr2tc object;
  expr2tc offset;
  unsigned int alignment;

  object_descriptor2t(
    const type2tc &t,
    const expr2tc &root,
    const expr2tc &offs,
    unsigned int align)
    : expr2t(t, object_descriptor_id),
      object(root),
      offset(offs),
      alignment(align)
  {
  }
  object_descriptor2t(const object_descriptor2t &ref) = default;

  const expr2tc &get_root_object() const;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &object_descriptor2t::object,
    &object_descriptor2t::offset,
    &object_descriptor2t::alignment);
  static std::string field_names[esbmct::num_type_fields];
};

class code_function_call2t : public expr2t
{
public:
  expr2tc ret;
  expr2tc function;
  std::vector<expr2tc> operands;
  locationt location; // not reflected: source loc travels with the stmt
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_function_call2t(
    const expr2tc &r,
    const expr2tc &func,
    const std::vector<expr2tc> &args,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_function_call_id),
      ret(r),
      function(func),
      operands(args),
      location(loc)
  {
  }
  code_function_call2t(const code_function_call2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &code_function_call2t::ret,
    &code_function_call2t::function,
    &code_function_call2t::operands);
  static std::string field_names[esbmct::num_type_fields];
};

// V.4 (esbmc/esbmc#4715): structured control-flow code kinds. IREP2 had only
// the flat goto-level code kinds; these mirror the legacy structured codet
// statements (ifthenelse/while/for/switch/break/continue/label) so the
// frontend can build IREP2 bodies and goto_convert can consume them, removing
// the per-instruction back-migration at the body seam (wall W1). Shipped
// dead-but-tested first: nothing builds them until the goto_convert wiring
// phase, so they are behaviour-inert and only the round-trip unit tests
// exercise them (the V-track pattern).
//
// V.4.1: each kind carries a `locationt location` so a future IREP2-native
// goto_convert can stamp each instruction's source location -- the legacy
// codet carries it on the node, but expr2t/migrate do not, so an IREP2 body
// would otherwise lose all counterexample line numbers. The field is
// deliberately NOT part of the `fields` tuple, so it does not enter the IREP2
// hash/equality (matching how a goto instructiont stores its locationt
// separately); it is preserved through clone() by the defaulted copy ctor and
// threaded by hand in migrate (forward copies code.location(), back restores
// it).
class code_ifthenelse2t : public expr2t
{
public:
  expr2tc cond;
  expr2tc then_case;
  expr2tc else_case;  // nil when there is no else branch
  locationt location; // not reflected (see note above)
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_ifthenelse2t(
    const expr2tc &c,
    const expr2tc &t,
    const expr2tc &e,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_ifthenelse_id),
      cond(c),
      then_case(t),
      else_case(e),
      location(loc)
  {
  }
  code_ifthenelse2t(const code_ifthenelse2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &code_ifthenelse2t::cond,
    &code_ifthenelse2t::then_case,
    &code_ifthenelse2t::else_case);
  static std::string field_names[esbmct::num_type_fields];
};

class code_while2t : public expr2t
{
public:
  expr2tc cond;
  expr2tc body;
  locationt location; // not reflected (see note above)
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_while2t(
    const expr2tc &c,
    const expr2tc &b,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_while_id), cond(c), body(b), location(loc)
  {
  }
  code_while2t(const code_while2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &code_while2t::cond, &code_while2t::body);
  static std::string field_names[esbmct::num_type_fields];
};

class code_dowhile2t : public expr2t
{
public:
  expr2tc cond;
  expr2tc body;
  locationt location; // not reflected (see note above)
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_dowhile2t(
    const expr2tc &c,
    const expr2tc &b,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_dowhile_id), cond(c), body(b), location(loc)
  {
  }
  code_dowhile2t(const code_dowhile2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &code_dowhile2t::cond,
    &code_dowhile2t::body);
  static std::string field_names[esbmct::num_type_fields];
};

class code_for2t : public expr2t
{
public:
  expr2tc init; // nil when absent
  expr2tc cond; // nil when absent
  expr2tc iter; // nil when absent
  expr2tc body;
  locationt location; // not reflected (see note above)
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_for2t(
    const expr2tc &i,
    const expr2tc &c,
    const expr2tc &it,
    const expr2tc &b,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_for_id),
      init(i),
      cond(c),
      iter(it),
      body(b),
      location(loc)
  {
  }
  code_for2t(const code_for2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &code_for2t::init,
    &code_for2t::cond,
    &code_for2t::iter,
    &code_for2t::body);
  static std::string field_names[esbmct::num_type_fields];
};

class code_switch2t : public expr2t
{
public:
  expr2tc value;
  expr2tc body;
  locationt location; // not reflected (see note above)
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_switch2t(
    const expr2tc &v,
    const expr2tc &b,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_switch_id), value(v), body(b), location(loc)
  {
  }
  code_switch2t(const code_switch2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &code_switch2t::value, &code_switch2t::body);
  static std::string field_names[esbmct::num_type_fields];
};

class code_break2t : public expr2t
{
public:
  locationt location; // not reflected (see note above)
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_break2t(const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_break_id), location(loc)
  {
  }
  code_break2t(const code_break2t &ref) = default;

  static constexpr auto fields = std::make_tuple(&expr2t::type);
  static std::string field_names[esbmct::num_type_fields];
};

class code_continue2t : public expr2t
{
public:
  locationt location; // not reflected (see note above)
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_continue2t(const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_continue_id), location(loc)
  {
  }
  code_continue2t(const code_continue2t &ref) = default;

  static constexpr auto fields = std::make_tuple(&expr2t::type);
  static std::string field_names[esbmct::num_type_fields];
};

class code_label2t : public expr2t
{
public:
  irep_idt label;
  expr2tc code;
  locationt location; // not reflected (see note above)
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_label2t(
    const irep_idt &l,
    const expr2tc &c,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_label_id), label(l), code(c), location(loc)
  {
  }
  code_label2t(const code_label2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &code_label2t::label, &code_label2t::code);
  static std::string field_names[esbmct::num_type_fields];
};

/** V.4.2: one case/default arm of a switch body. `is_default` is true for the
 *  default arm; `case_op` is nil in that case. `location` is not reflected
 *  (same pattern as the other V.4 kinds). */
class code_switch_case2t : public expr2t
{
public:
  bool is_default;
  expr2tc case_op; // nil when is_default
  expr2tc code;
  locationt location; // not reflected (see note above)
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_switch_case2t(
    bool _is_default,
    const expr2tc &_case_op,
    const expr2tc &_code,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_switch_case_id),
      is_default(_is_default),
      case_op(_case_op),
      code(_code),
      location(loc)
  {
  }
  code_switch_case2t(const code_switch_case2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &code_switch_case2t::is_default,
    &code_switch_case2t::case_op,
    &code_switch_case2t::code);
  static std::string field_names[esbmct::num_type_fields];
};

/** V.4.3: code_assert / code_assume — single-guard code kinds emitted by the
 *  Python / C++ frontends for assert and __ESBMC_assume.
 *  The guard is the boolean condition; the location carries the user-visible
 *  comment (assertion message) and source coordinates. */
class code_assert2t : public expr2t
{
public:
  expr2tc guard;
  locationt location; // not reflected (see note above)
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_assert2t(const expr2tc &g, const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_assert_id), guard(g), location(loc)
  {
  }
  code_assert2t(const code_assert2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &code_assert2t::guard);
  static std::string field_names[esbmct::num_type_fields];
};

class code_assume2t : public expr2t
{
public:
  expr2tc guard;
  locationt location; // not reflected (see note above)
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_assume2t(const expr2tc &g, const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_assume_id), guard(g), location(loc)
  {
  }
  code_assume2t(const code_assume2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &code_assume2t::guard);
  static std::string field_names[esbmct::num_type_fields];
};

/** V.4.2: wraps C-frontend sideeffect assignment nodes — simple `=` and
 *  compound `+=`, `-=`, etc. — for round-trip through migrate_expr /
 *  migrate_expr_back.  `op` carries the operator string exactly as it appears
 *  in the legacy irept (e.g. "assign", "assign+", "assign_div"), so that the
 *  back-migration can reconstruct the original sideeffect node verbatim. */
class sideeffect_assign2t : public expr2t
{
public:
  irep_idt op; // "assign", "assign+", "assign-", "assign*", etc.
  expr2tc lhs;
  expr2tc rhs;
  locationt location; // not reflected: source loc travels with the stmt
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  sideeffect_assign2t(
    const type2tc &t,
    const irep_idt &o,
    const expr2tc &l,
    const expr2tc &r,
    const locationt &loc = locationt())
    : expr2t(t, sideeffect_assign_id), op(o), lhs(l), rhs(r), location(loc)
  {
  }
  sideeffect_assign2t(const sideeffect_assign2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &sideeffect_assign2t::op,
    &sideeffect_assign2t::lhs,
    &sideeffect_assign2t::rhs);
  static std::string field_names[esbmct::num_type_fields];
};

class code_comma2t : public expr2t
{
public:
  expr2tc side_1;
  expr2tc side_2;

  code_comma2t(const type2tc &t, const expr2tc &s1, const expr2tc &s2)
    : expr2t(t, code_comma_id), side_1(s1), side_2(s2)
  {
  }
  code_comma2t(const code_comma2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &code_comma2t::side_1,
    &code_comma2t::side_2);
  static std::string field_names[esbmct::num_type_fields];
};

class invalid_pointer2t : public expr2t
{
public:
  expr2tc ptr_obj;

  invalid_pointer2t(const expr2tc &obj)
    : expr2t(get_bool_type(), invalid_pointer_id), ptr_obj(obj)
  {
  }
  invalid_pointer2t(const invalid_pointer2t &ref) = default;

  static constexpr auto fields = std::make_tuple(&invalid_pointer2t::ptr_obj);
  static std::string field_names[esbmct::num_type_fields];
};

class code_asm2t : public expr2t
{
public:
  irep_idt value;
  locationt location; // not reflected: source loc travels with the stmt
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_asm2t(
    const type2tc &type,
    const irep_idt &stringref,
    const locationt &loc = locationt())
    : expr2t(type, code_asm_id), value(stringref), location(loc)
  {
  }
  code_asm2t(const code_asm2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &code_asm2t::value);
  static std::string field_names[esbmct::num_type_fields];
};

class code_cpp_catch2t : public expr2t
{
public:
  std::vector<irep_idt> exception_list;
  // Source-level try/catch operands: operands[0] is the try block, operands
  // [1..N] the catch-handler blocks (parallel to exception_list). Empty for
  // the post-goto-convert CATCH-push/pop marker instructions, which carry only
  // the catchable-type list. Retained so a try/catch body survives the
  // --irep2-bodies round-trip (esbmc/esbmc#4715); convert_catch reads it back.
  std::vector<expr2tc> operands;
  locationt location; // not reflected: source loc travels with the stmt
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_cpp_catch2t(
    const std::vector<irep_idt> &el,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_cpp_catch_id),
      exception_list(el),
      location(loc)
  {
  }
  code_cpp_catch2t(
    const std::vector<irep_idt> &el,
    const std::vector<expr2tc> &ops,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_cpp_catch_id),
      exception_list(el),
      operands(ops),
      location(loc)
  {
  }
  code_cpp_catch2t(const code_cpp_catch2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &code_cpp_catch2t::exception_list,
    &code_cpp_catch2t::operands);
  static std::string field_names[esbmct::num_type_fields];
};

class code_cpp_throw2t : public expr2t
{
public:
  expr2tc operand;
  std::vector<irep_idt> exception_list;
  locationt location; // not reflected: source loc travels with the stmt
  static constexpr std::size_t excluded_field_bytes = sizeof(locationt);

  code_cpp_throw2t(
    const expr2tc &o,
    const std::vector<irep_idt> &l,
    const locationt &loc = locationt())
    : expr2t(get_empty_type(), code_cpp_throw_id),
      operand(o),
      exception_list(l),
      location(loc)
  {
  }
  code_cpp_throw2t(const code_cpp_throw2t &ref) = default;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &code_cpp_throw2t::operand,
    &code_cpp_throw2t::exception_list);
  static std::string field_names[esbmct::num_type_fields];
};

/** Bit concatenation of two operands. */
class concat2t : public expr2t
{
public:
  expr2tc side_1;
  expr2tc side_2;

  concat2t(const type2tc &type, const expr2tc &forward, const expr2tc &aft)
    : expr2t(type, concat_id), side_1(forward), side_2(aft)
  {
    assert(is_unsignedbv_type(forward));
    assert(is_unsignedbv_type(aft));
  }
  concat2t(const concat2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &concat2t::side_1, &concat2t::side_2);
  static std::string field_names[esbmct::num_type_fields];
};

class extract2t : public expr2t
{
public:
  expr2tc from;
  unsigned int upper;
  unsigned int lower;

  extract2t(
    const type2tc &type,
    const expr2tc &from_,
    unsigned int upper_,
    unsigned int lower_)
    : expr2t(type, extract_id), from(from_), upper(upper_), lower(lower_)
  {
  }
  extract2t(const extract2t &ref) = default;

  expr2tc do_simplify() const override;

  static constexpr auto fields = std::make_tuple(
    &expr2t::type,
    &extract2t::from,
    &extract2t::upper,
    &extract2t::lower);
  static std::string field_names[esbmct::num_type_fields];
};

// Same deal as for "type_macros": is_<name>2t predicates plus to_<name>2t
// downcasts routed through irep2_checked_expr_cast so a bad to_*2t throws
// irep2_cast_error in every build mode.
#define expr_macros(name)                                                      \
  inline bool is_##name##2t(const expr2tc &t)                                  \
  {                                                                            \
    return t->expr_id == expr2t::name##_id;                                    \
  }                                                                            \
  inline const name##2t & to_##name##2t(const expr2tc &t)                      \
  {                                                                            \
    return irep2_checked_expr_cast<const name##2t>(                            \
      *t, expr2t::name##_id, #name);                                           \
  }                                                                            \
  inline name##2t & to_##name##2t(expr2tc & t)                                 \
  {                                                                            \
    return irep2_checked_expr_cast<name##2t>(                                  \
      *t.get(), expr2t::name##_id, #name);                                     \
  }                                                                            \
  inline const name##2t * try_to_##name##2t(const expr2tc &t)                  \
  {                                                                            \
    return is_##name##2t(t) ? &to_##name##2t(t) : nullptr;                     \
  }

// Instantiate the is_/to_/try_to_ predicate triple for every kind in
// expr_kinds.inc. Same manifest as the enum and forward declarations
// above, so adding a kind is a single line there.
#define IREP2_EXPR(kind, pretty) expr_macros(kind)
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR

#undef expr_macros

#endif /* IREP2_EXPR_H_ */
