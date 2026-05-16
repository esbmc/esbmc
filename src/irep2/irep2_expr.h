#ifndef IREP2_EXPR_H_
#define IREP2_EXPR_H_

#include <util/config.h>
#include <util/c_types.h>
#include <util/fixedbv.h>
#include <util/ieee_float.h>
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

// Data definitions.

// X-macro field-list expanders.  The user passes a field-list macro
// that takes a per-field action `F` and a separator `S` and emits
//   F(type1, name1) S F(type2, name2) S ...
// We expand it several times inside ESBMC_DEFINE_DATA to generate
// declarations, ctor params, ctor init list, traits typedefs, and the
// trait-list passed to expr2t_traits.  Since the typedef action is
// inside the class body, &klass::n resolves through the implicit
// class context and we don't need to pass the class name through.
#define ESBMC_DATA_DECL(t, n) t n;
#define ESBMC_DATA_PARAM(t, n) const t &n##_arg
#define ESBMC_DATA_PARAM_MOVE(t, n) t n##_arg
#define ESBMC_DATA_INIT(t, n) n(n##_arg)
#define ESBMC_DATA_INIT_MOVE(t, n) n(std::move(n##_arg))
#define ESBMC_DATA_TRAIT_REF(t, n) n##_field
#define ESBMC_DATA_COMMA ,
#define ESBMC_DATA_NONE

#define ESBMC_DATA_TYPEDEF(t, n)                                               \
  typedef esbmct::field_traits<t, self_t, &self_t::n> n##_field;

/** Define a data class with the given full class name `klass`. Fields are
 *  passed via an X-macro `(F, S)`-style list of `(type, name)` pairs.
 *  Result type comes from the explicit `type2tc` ctor argument; the
 *  generated class extends expr2t directly.
 *
 *  Three flavours:
 *    - ESBMC_DEFINE_DATA_AS:        ctor takes fields by const ref,
 *                                   result type explicit, uses
 *                                   `expr2t_traits`.
 *    - ESBMC_DEFINE_DATA_NOTYPE_AS: ctor takes fields by const ref,
 *                                   result type implicit, uses
 *                                   `expr2t_traits_notype`.
 *    - ESBMC_DEFINE_DATA_MOVE_AS:   ctor takes fields by value and
 *                                   moves them (use for std::vector
 *                                   and similar heavy types).
 *
 *  Convenience aliases below suffix `klass` with `_data` automatically. */
#define ESBMC_DEFINE_DATA_AS(klass, FIELDS)                                    \
  class klass : public expr2t                                                  \
  {                                                                            \
  public:                                                                      \
    using self_t = klass;                                                      \
    klass(                                                                     \
      const type2tc &t,                                                        \
      expr2t::expr_ids id,                                                     \
      FIELDS(ESBMC_DATA_PARAM, ESBMC_DATA_COMMA))                              \
      : expr2t(t, id), FIELDS(ESBMC_DATA_INIT, ESBMC_DATA_COMMA)               \
    {                                                                          \
    }                                                                          \
    klass(const klass &ref) = default;                                         \
    FIELDS(ESBMC_DATA_DECL, ESBMC_DATA_NONE)                                   \
    FIELDS(ESBMC_DATA_TYPEDEF, ESBMC_DATA_NONE)                                \
    typedef esbmct::expr2t_traits<                                             \
      FIELDS(ESBMC_DATA_TRAIT_REF, ESBMC_DATA_COMMA)>                          \
      traits;                                                                  \
  }

#define ESBMC_DEFINE_DATA_NOTYPE_AS(klass, FIELDS)                             \
  class klass : public expr2t                                                  \
  {                                                                            \
  public:                                                                      \
    using self_t = klass;                                                      \
    klass(                                                                     \
      const type2tc &t,                                                        \
      expr2t::expr_ids id,                                                     \
      FIELDS(ESBMC_DATA_PARAM, ESBMC_DATA_COMMA))                              \
      : expr2t(t, id), FIELDS(ESBMC_DATA_INIT, ESBMC_DATA_COMMA)               \
    {                                                                          \
    }                                                                          \
    klass(const klass &ref) = default;                                         \
    FIELDS(ESBMC_DATA_DECL, ESBMC_DATA_NONE)                                   \
    FIELDS(ESBMC_DATA_TYPEDEF, ESBMC_DATA_NONE)                                \
    typedef esbmct::expr2t_traits_notype<                                      \
      FIELDS(ESBMC_DATA_TRAIT_REF, ESBMC_DATA_COMMA)>                          \
      traits;                                                                  \
  }

#define ESBMC_DEFINE_DATA_MOVE_AS(klass, FIELDS)                               \
  class klass : public expr2t                                                  \
  {                                                                            \
  public:                                                                      \
    using self_t = klass;                                                      \
    klass(                                                                     \
      const type2tc &t,                                                        \
      expr2t::expr_ids id,                                                     \
      FIELDS(ESBMC_DATA_PARAM_MOVE, ESBMC_DATA_COMMA))                         \
      : expr2t(t, id), FIELDS(ESBMC_DATA_INIT_MOVE, ESBMC_DATA_COMMA)          \
    {                                                                          \
    }                                                                          \
    klass(const klass &ref) = default;                                         \
    FIELDS(ESBMC_DATA_DECL, ESBMC_DATA_NONE)                                   \
    FIELDS(ESBMC_DATA_TYPEDEF, ESBMC_DATA_NONE)                                \
    typedef esbmct::expr2t_traits<                                             \
      FIELDS(ESBMC_DATA_TRAIT_REF, ESBMC_DATA_COMMA)>                          \
      traits;                                                                  \
  }

#define ESBMC_DEFINE_DATA(name, FIELDS)                                        \
  ESBMC_DEFINE_DATA_AS(name##_data, FIELDS)
#define ESBMC_DEFINE_DATA_NOTYPE(name, FIELDS)                                 \
  ESBMC_DEFINE_DATA_NOTYPE_AS(name##_data, FIELDS)
#define ESBMC_DEFINE_DATA_MOVE(name, FIELDS)                                   \
  ESBMC_DEFINE_DATA_MOVE_AS(name##_data, FIELDS)

/** Define a data class that extends an existing one (`parent`) and adds
 *  more fields. `PARENT_FIELDS` is the parent's X-macro list (re-passed
 *  so the ctor can take the parent's args by value/move and forward
 *  them).  `OWN_FIELDS` lists this class's own additional fields.  The
 *  parent's trait typedefs are referenced via name##_field (they are
 *  inherited from the parent class scope). */
#define ESBMC_DEFINE_DATA_EXTENDS(name, parent, PARENT_FIELDS, OWN_FIELDS)     \
  class name##_data : public parent                                            \
  {                                                                            \
  public:                                                                      \
    using self_t = name##_data;                                                \
    name##_data(                                                               \
      const type2tc &t,                                                        \
      expr2t::expr_ids id,                                                     \
      PARENT_FIELDS(ESBMC_DATA_PARAM_MOVE, ESBMC_DATA_COMMA),                  \
      OWN_FIELDS(ESBMC_DATA_PARAM, ESBMC_DATA_COMMA))                          \
      : parent(t, id, PARENT_FIELDS(ESBMC_DATA_FWD, ESBMC_DATA_COMMA)),        \
        OWN_FIELDS(ESBMC_DATA_INIT, ESBMC_DATA_COMMA)                          \
    {                                                                          \
    }                                                                          \
    name##_data(const name##_data &ref) = default;                             \
    OWN_FIELDS(ESBMC_DATA_DECL, ESBMC_DATA_NONE)                               \
    OWN_FIELDS(ESBMC_DATA_TYPEDEF, ESBMC_DATA_NONE)                            \
    typedef esbmct::expr2t_traits<                                             \
      PARENT_FIELDS(ESBMC_DATA_TRAIT_REF, ESBMC_DATA_COMMA),                   \
      OWN_FIELDS(ESBMC_DATA_TRAIT_REF, ESBMC_DATA_COMMA)>                      \
      traits;                                                                  \
  }

// Forward the parent's argument as-is (already a movable rvalue).
#define ESBMC_DATA_FWD(t, n) std::move(n##_arg)

#define ESBMC_FIELDS_constant_int(F, S) F(BigInt, value)
ESBMC_DEFINE_DATA(constant_int, ESBMC_FIELDS_constant_int);

#define ESBMC_FIELDS_constant_fixedbv(F, S) F(fixedbvt, value)
ESBMC_DEFINE_DATA(constant_fixedbv, ESBMC_FIELDS_constant_fixedbv);

#define ESBMC_FIELDS_constant_floatbv(F, S) F(ieee_floatt, value)
ESBMC_DEFINE_DATA(constant_floatbv, ESBMC_FIELDS_constant_floatbv);

#define ESBMC_FIELDS_dereference(F, S) F(expr2tc, value)
ESBMC_DEFINE_DATA(dereference, ESBMC_FIELDS_dereference);

#define ESBMC_FIELDS_bitcast(F, S) F(expr2tc, from)
ESBMC_DEFINE_DATA(bitcast, ESBMC_FIELDS_bitcast);

#define ESBMC_FIELDS_member_ref(F, S) F(irep_idt, member)
ESBMC_DEFINE_DATA(member_ref, ESBMC_FIELDS_member_ref);

#define ESBMC_FIELDS_code_decl(F, S) F(irep_idt, value)
ESBMC_DEFINE_DATA(code_decl, ESBMC_FIELDS_code_decl);

#define ESBMC_FIELDS_code_goto(F, S) F(irep_idt, target)
ESBMC_DEFINE_DATA(code_goto, ESBMC_FIELDS_code_goto);

#define ESBMC_FIELDS_code_asm(F, S) F(irep_idt, value)
ESBMC_DEFINE_DATA(code_asm, ESBMC_FIELDS_code_asm);

#define ESBMC_FIELDS_same_object(F, S) F(expr2tc, side_1) S F(expr2tc, side_2)
ESBMC_DEFINE_DATA(same_object, ESBMC_FIELDS_same_object);

#define ESBMC_FIELDS_code_assign(F, S) F(expr2tc, target) S F(expr2tc, source)
ESBMC_DEFINE_DATA(code_assign, ESBMC_FIELDS_code_assign);

#define ESBMC_FIELDS_code_comma(F, S) F(expr2tc, side_1) S F(expr2tc, side_2)
ESBMC_DEFINE_DATA(code_comma, ESBMC_FIELDS_code_comma);

#define ESBMC_FIELDS_constant_bool(F, S) F(bool, value)
ESBMC_DEFINE_DATA(constant_bool, ESBMC_FIELDS_constant_bool);

#define ESBMC_FIELDS_constant_array_of(F, S) F(expr2tc, initializer)
ESBMC_DEFINE_DATA(constant_array_of, ESBMC_FIELDS_constant_array_of);

#define ESBMC_FIELDS_constant_datatype(F, S)                                   \
  F(std::vector<expr2tc>, datatype_members)
ESBMC_DEFINE_DATA_MOVE(constant_datatype, ESBMC_FIELDS_constant_datatype);

#define ESBMC_FIELDS_code_block(F, S) F(std::vector<expr2tc>, operands)
ESBMC_DEFINE_DATA_MOVE(code_block, ESBMC_FIELDS_code_block);

#define ESBMC_FIELDS_code_expression(F, S) F(expr2tc, operand)
ESBMC_DEFINE_DATA_NOTYPE(code_expression, ESBMC_FIELDS_code_expression);

#define ESBMC_FIELDS_code_cpp_catch(F, S)                                      \
  F(std::vector<irep_idt>, exception_list)
ESBMC_DEFINE_DATA_MOVE(code_cpp_catch, ESBMC_FIELDS_code_cpp_catch);

#define ESBMC_FIELDS_typecast(F, S) F(expr2tc, from) S F(expr2tc, rounding_mode)
ESBMC_DEFINE_DATA(typecast, ESBMC_FIELDS_typecast);

#define ESBMC_FIELDS_if(F, S)                                                  \
  F(expr2tc, cond) S F(expr2tc, true_value) S F(expr2tc, false_value)
ESBMC_DEFINE_DATA(if, ESBMC_FIELDS_if);

#define ESBMC_FIELDS_relation(F, S) F(expr2tc, side_1) S F(expr2tc, side_2)
ESBMC_DEFINE_DATA(relation, ESBMC_FIELDS_relation);

/** Like `constant_datatype_data` but tags the active union variant
 *  (`init_field`). Inherits `datatype_members` from its parent. */
#define ESBMC_FIELDS_constant_union_own(F, S) F(irep_idt, init_field)
ESBMC_DEFINE_DATA_EXTENDS(
  constant_union,
  constant_datatype_data,
  ESBMC_FIELDS_constant_datatype,
  ESBMC_FIELDS_constant_union_own);

/** Kind of string literal carried by a `constant_string2t`.  Lives at
 *  namespace scope (rather than inside the data class) so the X-macro
 *  fold can mention it as a plain type name. */
enum class constant_string_kindt
{
  DEFAULT, /* "" */
  WIDE,    /* L"" */
  UNICODE, /* u8"", u"" and U"" */
};

#define ESBMC_FIELDS_constant_string(F, S)                                     \
  F(irep_idt, value) S F(constant_string_kindt, kind)
ESBMC_DEFINE_DATA(constant_string, ESBMC_FIELDS_constant_string);

/** Symex renaming level.  Lives at namespace scope (rather than inside
 *  symbol_data) so the X-macro fold can name it as a plain type.
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

#define ESBMC_FIELDS_symbol(F, S)                                              \
  F(irep_idt, thename)                                                         \
  S F(symbol_renaming_level, rlevel) S F(unsigned int, level1_num) S F(        \
    unsigned int, level2_num) S                                                \
  F(unsigned int, thread_num) S                                                \
  F(unsigned int, node_num)
ESBMC_DEFINE_DATA(symbol, ESBMC_FIELDS_symbol);

#define ESBMC_FIELDS_value_only(F, S) F(expr2tc, value)
ESBMC_DEFINE_DATA_NOTYPE_AS(bool_1op, ESBMC_FIELDS_value_only);
ESBMC_DEFINE_DATA_AS(arith_1op, ESBMC_FIELDS_value_only);

#define ESBMC_FIELDS_two_sides(F, S) F(expr2tc, side_1) S F(expr2tc, side_2)
ESBMC_DEFINE_DATA_AS(logic_2ops, ESBMC_FIELDS_two_sides);
ESBMC_DEFINE_DATA_AS(bit_2ops, ESBMC_FIELDS_two_sides);

ESBMC_DEFINE_DATA_AS(arith_2ops, ESBMC_FIELDS_two_sides);

/** Debug-only consistency check for arith_2ops operands and result type.
 *  Validates that pointer-difference, bv-vs-bv arithmetic and pointer-bv
 *  arithmetic have matching shapes. Called from the concrete arith
 *  classes (add2t/sub2t/...) under NDEBUG; a no-op in Release. */
void assert_arith_2ops_consistency(
  const type2tc &t,
  expr2t::expr_ids id,
  const expr2tc &v1,
  const expr2tc &v2);

#define ESBMC_FIELDS_ieee_1(F, S) F(expr2tc, rounding_mode) S F(expr2tc, value)
ESBMC_DEFINE_DATA_AS(ieee_arith_1op, ESBMC_FIELDS_ieee_1);

#define ESBMC_FIELDS_ieee_2(F, S)                                              \
  F(expr2tc, rounding_mode) S F(expr2tc, side_1) S F(expr2tc, side_2)
ESBMC_DEFINE_DATA_AS(ieee_arith_2ops, ESBMC_FIELDS_ieee_2);

#define ESBMC_FIELDS_ieee_3(F, S)                                              \
  F(expr2tc, rounding_mode)                                                    \
  S F(expr2tc, value_1) S F(expr2tc, value_2) S F(expr2tc, value_3)
ESBMC_DEFINE_DATA_AS(ieee_arith_3ops, ESBMC_FIELDS_ieee_3);

#define ESBMC_FIELDS_ptr_obj(F, S) F(expr2tc, ptr_obj)
ESBMC_DEFINE_DATA_AS(pointer_ops, ESBMC_FIELDS_ptr_obj);

// Special class for invalid_pointer2t, which needs always-construct
// forcing.  Storage matches pointer_ops but the traits are notype.
ESBMC_DEFINE_DATA_NOTYPE_AS(invalid_pointer_ops, ESBMC_FIELDS_ptr_obj);

#define ESBMC_FIELDS_byte_extract(F, S)                                        \
  F(expr2tc, source_value) S F(expr2tc, source_offset) S F(bool, big_endian)
ESBMC_DEFINE_DATA(byte_extract, ESBMC_FIELDS_byte_extract);

#define ESBMC_FIELDS_byte_update(F, S)                                         \
  F(expr2tc, source_value)                                                     \
  S F(expr2tc, source_offset) S F(expr2tc, update_value) S F(bool, big_endian)
ESBMC_DEFINE_DATA(byte_update, ESBMC_FIELDS_byte_update);

#define ESBMC_FIELDS_with(F, S)                                                \
  F(expr2tc, source_value) S F(expr2tc, update_field) S F(expr2tc, update_value)
ESBMC_DEFINE_DATA(with, ESBMC_FIELDS_with);

#define ESBMC_FIELDS_member(F, S) F(expr2tc, source_value) S F(irep_idt, member)
ESBMC_DEFINE_DATA(member, ESBMC_FIELDS_member);

#define ESBMC_FIELDS_ptr_mem(F, S)                                             \
  F(expr2tc, source_value) S F(expr2tc, member_pointer)
ESBMC_DEFINE_DATA(ptr_mem, ESBMC_FIELDS_ptr_mem);

#define ESBMC_FIELDS_index(F, S) F(expr2tc, source_value) S F(expr2tc, index)
ESBMC_DEFINE_DATA(index, ESBMC_FIELDS_index);

#define ESBMC_FIELDS_string_ops(F, S) F(expr2tc, string)
ESBMC_DEFINE_DATA_AS(string_ops, ESBMC_FIELDS_string_ops);

#define ESBMC_FIELDS_overflow_ops(F, S) F(expr2tc, operand)
ESBMC_DEFINE_DATA_AS(overflow_ops, ESBMC_FIELDS_overflow_ops);

// overflow_cast_data extends overflow_ops with a `bits` field.
#define ESBMC_FIELDS_overflow_cast_own(F, S) F(unsigned int, bits)
ESBMC_DEFINE_DATA_EXTENDS(
  overflow_cast,
  overflow_ops,
  ESBMC_FIELDS_overflow_ops,
  ESBMC_FIELDS_overflow_cast_own);

#define ESBMC_FIELDS_dynamic_object(F, S)                                      \
  F(expr2tc, instance) S F(bool, invalid) S F(bool, unknown)
ESBMC_DEFINE_DATA(dynamic_object, ESBMC_FIELDS_dynamic_object);

ESBMC_DEFINE_DATA_NOTYPE_AS(object_ops, ESBMC_FIELDS_value_only);

/** Enumeration identifying each particular kind of side effect. Lifted to
 *  namespace scope so the X-macro fold can name it as a plain type. */
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
  old_snapshot,  // For __ESBMC_old() in function contracts
  assigns_target // For __ESBMC_assigns() in function contracts
};

#define ESBMC_FIELDS_sideeffect(F, S)                                          \
  F(expr2tc, operand)                                                          \
  S F(expr2tc, size) S F(std::vector<expr2tc>, arguments)                      \
    S F(type2tc, alloctype) S F(sideeffect_allockind, kind)
ESBMC_DEFINE_DATA_MOVE(sideeffect, ESBMC_FIELDS_sideeffect);

/** Which member of the printf family a `code_printf2t` represents.  The
 *  symex side (src/goto-symex/builtin_functions/io.cpp) switches on this
 *  to pick the correct argument layout. */
enum class printf_kindt
{
  PRINTF,
  FPRINTF,
  DPRINTF,
  SPRINTF,
  VFPRINTF,
  SNPRINTF,
};

/** Maps the textual base_name of a printf-family symbol (e.g. "printf",
 *  "snprintf") onto a printf_kindt.  Aborts on an unknown name. */
printf_kindt printf_kind_from_name(const irep_idt &name);

#define ESBMC_FIELDS_code_printf(F, S)                                         \
  F(std::vector<expr2tc>, operands) S F(printf_kindt, kind)
ESBMC_DEFINE_DATA_MOVE(code_printf, ESBMC_FIELDS_code_printf);

#define ESBMC_FIELDS_object_desc(F, S)                                         \
  F(expr2tc, object) S F(expr2tc, offset) S F(unsigned int, alignment)
ESBMC_DEFINE_DATA(object_desc, ESBMC_FIELDS_object_desc);

#define ESBMC_FIELDS_code_funccall(F, S)                                       \
  F(expr2tc, ret) S F(expr2tc, function) S F(std::vector<expr2tc>, operands)
ESBMC_DEFINE_DATA_MOVE(code_funccall, ESBMC_FIELDS_code_funccall);

#define ESBMC_FIELDS_code_cpp_throw(F, S)                                      \
  F(expr2tc, operand) S F(std::vector<irep_idt>, exception_list)
ESBMC_DEFINE_DATA_MOVE(code_cpp_throw, ESBMC_FIELDS_code_cpp_throw);

#define ESBMC_FIELDS_code_cpp_throw_decl(F, S)                                 \
  F(std::vector<irep_idt>, exception_list)
ESBMC_DEFINE_DATA_MOVE(code_cpp_throw_decl, ESBMC_FIELDS_code_cpp_throw_decl);

#define ESBMC_FIELDS_extract(F, S)                                             \
  F(expr2tc, from) S F(unsigned int, upper) S F(unsigned int, lower)
ESBMC_DEFINE_DATA(extract, ESBMC_FIELDS_extract);

// Give everything a typedef name. Use this to construct both the templated
// expression methods, but also the container class which needs the template
// parameters too.
// Given how otherwise this means typing a large amount of template arguments
// again and again, this gets macro'd.

#define irep_typedefs(basename, superclass)                                    \
  template <typename... Args>                                                  \
  inline expr2tc basename##2tc(Args && ...args)                                \
  {                                                                            \
    return make_irep<basename##2t>(std::forward<Args>(args)...);               \
  }                                                                            \
  typedef esbmct::expr_methods2<basename##2t, superclass, superclass::traits>  \
    basename##_expr_methods;                                                   \
  extern template class esbmct::                                               \
    expr_methods2<basename##2t, superclass, superclass::traits>;               \
  extern template class esbmct::                                               \
    irep_methods2<basename##2t, superclass, superclass::traits>;

// This can't be replaced by iterating over all expr ids in preprocessing
// magic because the mapping between top level expr class and it's data holding
// object isn't regular: the data class depends on /what/ the expression /is/.
irep_typedefs(constant_int, constant_int_data);
irep_typedefs(constant_fixedbv, constant_fixedbv_data);
irep_typedefs(constant_floatbv, constant_floatbv_data);
irep_typedefs(constant_struct, constant_datatype_data);
irep_typedefs(constant_union, constant_union_data);
irep_typedefs(constant_array, constant_datatype_data);
irep_typedefs(constant_vector, constant_datatype_data);
irep_typedefs(constant_bool, constant_bool_data);
irep_typedefs(constant_array_of, constant_array_of_data);
irep_typedefs(constant_string, constant_string_data);
irep_typedefs(symbol, symbol_data);
irep_typedefs(nearbyint, typecast_data);
irep_typedefs(typecast, typecast_data);
irep_typedefs(bitcast, bitcast_data);
irep_typedefs(if, if_data);
irep_typedefs(equality, relation_data);
irep_typedefs(notequal, relation_data);
irep_typedefs(lessthan, relation_data);
irep_typedefs(greaterthan, relation_data);
irep_typedefs(lessthanequal, relation_data);
irep_typedefs(greaterthanequal, relation_data);
irep_typedefs(cmp_three_way, relation_data);
irep_typedefs(not, bool_1op);
irep_typedefs(and, logic_2ops);
irep_typedefs(or, logic_2ops);
irep_typedefs(xor, logic_2ops);
irep_typedefs(implies, logic_2ops);
irep_typedefs(bitand, bit_2ops);
irep_typedefs(bitor, bit_2ops);
irep_typedefs(bitxor, bit_2ops);
irep_typedefs(lshr, bit_2ops);
irep_typedefs(bitnot, arith_1op);
irep_typedefs(neg, arith_1op);
irep_typedefs(abs, arith_1op);
irep_typedefs(add, arith_2ops);
irep_typedefs(sub, arith_2ops);
irep_typedefs(mul, arith_2ops);
irep_typedefs(div, arith_2ops);
irep_typedefs(ieee_add, ieee_arith_2ops);
irep_typedefs(ieee_sub, ieee_arith_2ops);
irep_typedefs(ieee_mul, ieee_arith_2ops);
irep_typedefs(ieee_div, ieee_arith_2ops);
irep_typedefs(ieee_fma, ieee_arith_3ops);
irep_typedefs(ieee_sqrt, ieee_arith_1op);
irep_typedefs(modulus, arith_2ops);
irep_typedefs(shl, bit_2ops);
irep_typedefs(ashr, bit_2ops);
irep_typedefs(same_object, same_object_data);
irep_typedefs(pointer_offset, pointer_ops);
irep_typedefs(pointer_object, pointer_ops);
irep_typedefs(pointer_capability, pointer_ops);
irep_typedefs(address_of, pointer_ops);
irep_typedefs(byte_extract, byte_extract_data);
irep_typedefs(byte_update, byte_update_data);
irep_typedefs(with, with_data);
irep_typedefs(member, member_data);
irep_typedefs(member_ref, member_ref_data);
irep_typedefs(ptr_mem, ptr_mem_data);
irep_typedefs(index, index_data);
irep_typedefs(isnan, bool_1op);
irep_typedefs(overflow, overflow_ops);
irep_typedefs(overflow_cast, overflow_cast_data);
irep_typedefs(overflow_neg, overflow_ops);
irep_typedefs(unknown, expr2t);
irep_typedefs(invalid, expr2t);
irep_typedefs(null_object, expr2t);
irep_typedefs(dynamic_object, dynamic_object_data);
irep_typedefs(dereference, dereference_data);
irep_typedefs(valid_object, object_ops);
irep_typedefs(races_check, object_ops);
irep_typedefs(deallocated_obj, object_ops);
irep_typedefs(dynamic_size, object_ops);
irep_typedefs(sideeffect, sideeffect_data);
irep_typedefs(code_block, code_block_data);
irep_typedefs(code_assign, code_assign_data);
irep_typedefs(code_decl, code_decl_data);
irep_typedefs(code_dead, code_decl_data);
irep_typedefs(code_printf, code_printf_data);
irep_typedefs(code_expression, code_expression_data);
irep_typedefs(code_return, code_expression_data);
irep_typedefs(code_skip, expr2t);
irep_typedefs(code_free, code_expression_data);
irep_typedefs(code_goto, code_goto_data);
irep_typedefs(object_descriptor, object_desc_data);
irep_typedefs(code_function_call, code_funccall_data);
irep_typedefs(code_comma, code_comma_data);
irep_typedefs(invalid_pointer, invalid_pointer_ops);
irep_typedefs(code_asm, code_asm_data);
irep_typedefs(code_cpp_del_array, code_expression_data);
irep_typedefs(code_cpp_delete, code_expression_data);
irep_typedefs(code_cpp_catch, code_cpp_catch_data);
irep_typedefs(code_cpp_throw, code_cpp_throw_data);
irep_typedefs(code_cpp_throw_decl, code_cpp_throw_decl_data);
irep_typedefs(code_cpp_throw_decl_end, code_cpp_throw_decl_data);
irep_typedefs(isinf, bool_1op);
irep_typedefs(isnormal, bool_1op);
irep_typedefs(isfinite, bool_1op);
irep_typedefs(signbit, overflow_ops);
irep_typedefs(popcount, overflow_ops);
irep_typedefs(bswap, arith_1op);
irep_typedefs(concat, bit_2ops);
irep_typedefs(extract, extract_data);
irep_typedefs(capability_base, object_ops);
irep_typedefs(capability_top, object_ops);
irep_typedefs(forall, logic_2ops);
irep_typedefs(exists, logic_2ops);
irep_typedefs(isinstance, logic_2ops);
irep_typedefs(hasattr, logic_2ops);
irep_typedefs(isnone, logic_2ops);

class exists2t : public exists_expr_methods
{
public:
  exists2t(const type2tc &type, const expr2tc &sym, const expr2tc &predicate)
    : exists_expr_methods(type, exists_id, sym, predicate)
  {
  }
  exists2t(const exists2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class forall2t : public forall_expr_methods
{
public:
  forall2t(const type2tc &type, const expr2tc &sym, const expr2tc &predicate)
    : forall_expr_methods(type, forall_id, sym, predicate)
  {
  }
  forall2t(const forall2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

/** Constant integer class.
 *  Records a constant integer of an arbitary precision, signed or unsigned.
 *  Simplification operations will cause the integer to be clipped to whatever
 *  bit size is in expr type.
 *  @extends constant_int_data
 */
class constant_int2t : public constant_int_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of this integer.
   *  @param input BigInt object containing the integer we're dealing with
   */
  constant_int2t(const type2tc &type, const BigInt &input)
    : constant_int_expr_methods(type, constant_int_id, input)
  {
  }
  constant_int2t(const constant_int2t &ref) = default;

  /** Accessor for fetching machine-word unsigned integer of this constant */
  unsigned long as_ulong() const;
  /** Accessor for fetching machine-word integer of this constant */
  long as_long() const;

  static std::string field_names[esbmct::num_type_fields];
};

/** Constant fixedbv class. Records a fixed-width number in what I assume
 *  to be mantissa/exponent form, but which is described throughout CBMC code
 *  as fraction/integer parts. Stored in a fixedbvt.
 *  @extends constant_fixedbv_data
 */
class constant_fixedbv2t : public constant_fixedbv_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of this expression.
   *  @param value fixedbvt object containing number we'll be operating on
   */
  constant_fixedbv2t(const fixedbvt &value)
    : constant_fixedbv_expr_methods(
        value.spec.get_type(),
        constant_fixedbv_id,
        value)
  {
  }
  constant_fixedbv2t(const constant_fixedbv2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

/** Constant floatbv class. Records a floating-point number,
 *  Stored in a ieee_floatt.
 *  @extends constant_floatbv_data
 */
class constant_floatbv2t : public constant_floatbv_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of this expression.
   *  @param value ieee_floatt object containing number we'll be operating on
   */
  constant_floatbv2t(const ieee_floatt &value)
    : constant_floatbv_expr_methods(
        value.spec.get_type(),
        constant_floatbv_id,
        value)
  {
  }
  constant_floatbv2t(const constant_floatbv2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

/** Constant boolean value.
 *  Contains a constant bool; rather self explanatory.
 *  @extends constant_bool_data
 */
class constant_bool2t : public constant_bool_expr_methods
{
public:
  /** Primary constructor. @param value True or false */
  constant_bool2t(bool value)
    : constant_bool_expr_methods(get_bool_type(), constant_bool_id, value)
  {
  }
  constant_bool2t(const constant_bool2t &ref) = default;

  /** Return whether contained boolean is true. */
  bool is_true() const;
  /** Return whether contained boolean is false. */
  bool is_false() const;

  static std::string field_names[esbmct::num_type_fields];
};

/** Constant class for string constants.
 *  Contains an irep_idt representing the constant string.
 *  @extends constant_string_data
 */
class constant_string2t : public constant_string_expr_methods
{
public:
  using kindt = constant_string_kindt;

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
    : constant_string_expr_methods(type, constant_string_id, stringref, kind)
  {
  }
  constant_string2t(const constant_string2t &ref) = default;

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
 *  @extends constant_datatype_data
 */
class constant_struct2t : public constant_struct_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of this structure, presumably a struct_type2t
   *  @param membrs Vector of member values that make up this struct.
   */
  constant_struct2t(const type2tc &type, const std::vector<expr2tc> &members)
    : constant_struct_expr_methods(type, constant_struct_id, members)
  {
  }
  constant_struct2t(const constant_struct2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Constant union expression.
 *  Almost the same as constant_struct2t - a vector of members corresponding
 *  to the members described in the type. However, it seems the values pumped
 *  at us by CBMC only ever have one member (at position 0) representing the
 *  most recent value written to the union.
 *  @extend constant_union_data
 */
class constant_union2t : public constant_union_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of this structure, presumably a union_type2t
   *  @param members Vector of member values that make up this union.
   */
  constant_union2t(
    const type2tc &type,
    irep_idt init_field,
    const std::vector<expr2tc> &members)
    : constant_union_expr_methods(type, constant_union_id, members, init_field)
  {
    assert(is_union_type(type));
    // smt_conv.cpp's counterexample reconstruction intentionally builds unions
    //  with multiple members (see TODO in get_by_ast), so we can't check if the
    // union has at most 1 member initializer, with
    // assert(this->datatype_members.size() <= 1);
  }
  constant_union2t(const constant_union2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

/** Constant array.
 *  Contains a vector of array elements, pretty self explanatory. Only valid if
 *  its type has a constant sized array, can't have constant arrays of dynamic
 *  or infinitely sized arrays.
 *  @extends constant_datatype_data
 */
class constant_array2t : public constant_array_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of this array, must be a constant sized array
   *  @param membrs Vector of elements in this array
   */
  constant_array2t(const type2tc &type, const std::vector<expr2tc> &members)
    : constant_array_expr_methods(type, constant_array_id, members)
  {
    assert(type->type_id == type2t::array_id);
  }
  constant_array2t(const constant_array2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Constant array.
 *  Contains a vector of array elements, pretty self explanatory. Only valid if
 *  its type has a constant sized array, can't have constant arrays of dynamic
 *  or infinitely sized arrays.
 *  @extends constant_datatype_data
 */
class constant_vector2t : public constant_vector_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of this array, must be a constant sized array
   *  @param membrs Vector of elements in this array
   */
  constant_vector2t(const type2tc &type, const std::vector<expr2tc> &members)
    : constant_vector_expr_methods(type, constant_vector_id, members)
  {
  }
  constant_vector2t(const constant_vector2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

/** Constant array of one particular value.
 *  Expression with array type, possibly dynamic or infinitely sized, with
 *  all elements initialized to a single value.
 *  @extends constant_array_of_data
 */
class constant_array_of2t : public constant_array_of_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of this expression, must be an array.
   *  @param init Initializer for each element in this array
   */
  constant_array_of2t(const type2tc &type, const expr2tc &init)
    : constant_array_of_expr_methods(type, constant_array_of_id, init)
  {
  }
  constant_array_of2t(const constant_array_of2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

/** Symbol type.
 *  Contains the name of some variable. Various levels of renaming.
 *  @extends symbol_data
 */
class symbol2t : public symbol_expr_methods
{
public:
  using renaming_level = symbol_renaming_level;

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
    : symbol_expr_methods(type, symbol_id, init, lev, l1, l2, trd, node)
  {
    /* At some point in the past, symbols named "NULL" and "0" were equivalent.
     * The symbol called "0" should no longer be created for uniformity reasons.
     * Confirm that here, since support for it has been removed from smt_convt.
     * No other reason to disallow "0" as a symbol. */
    assert(init != "0");
  }

  symbol2t(const symbol2t &ref) = default;

  std::string get_symbol_name() const;

  static std::string field_names[esbmct::num_type_fields];
};

/** Nearbyint expression.
 *  Represents a rounding operation on a floatbv, we extend typecast as
 *  it already have a field for the rounding mode
 *  @extends typecast_data
 */
class nearbyint2t : public nearbyint_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type to round to
   *  @param from Expression to round from.
   *  @param rounding_mode Rounding mode, important only for floatbvs
   */
  nearbyint2t(
    const type2tc &type,
    const expr2tc &from,
    const expr2tc &rounding_mode)
    : nearbyint_expr_methods(type, nearbyint_id, from, rounding_mode)
  {
  }

  /** Primary constructor. This constructor defaults the rounding mode to
   *  the __ESBMC_rounding_mode symbol
   *  @param type Type to round to
   *  @param from Expression to round from.
   */
  nearbyint2t(const type2tc &type, const expr2tc &from)
    : nearbyint_expr_methods(
        type,
        nearbyint_id,
        from,
        symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode"))
  {
  }

  nearbyint2t(const nearbyint2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Typecast expression.
 *  Represents cast from contained expression 'from' to the type of this
 *  typecast.
 *  @extends typecast_data
 */
class typecast2t : public typecast_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type to typecast to
   *  @param from Expression to cast from.
   *  @param rounding_mode Rounding mode, important only for floatbvs
   */
  typecast2t(
    const type2tc &type,
    const expr2tc &from,
    const expr2tc &rounding_mode)
    : typecast_expr_methods(type, typecast_id, from, rounding_mode)
  {
  }

  /** Primary constructor. This constructor defaults the rounding mode to
   *  the __ESBMC_rounding_mode symbol
   *  @param type Type to typecast to
   *  @param from Expression to cast from.
   */
  typecast2t(const type2tc &type, const expr2tc &from)
    : typecast_expr_methods(
        type,
        typecast_id,
        from,
        symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode"))
  {
  }

  typecast2t(const typecast2t &ref) = default;
  expr2tc do_simplify() const override;

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
 *  @extends bitcast_data
 */
class bitcast2t : public bitcast_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type to bitcast to
   *  @param from Expression to cast from.
   */
  bitcast2t(const type2tc &type, const expr2tc &from)
    : bitcast_expr_methods(type, bitcast_id, from)
  {
    try
    {
      assert(type->get_width() == from->type->get_width());
    }
    catch (const type2t::symbolic_type_excp &)
    {
      /* ignore */
    }
  }

  bitcast2t(const bitcast2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** If-then-else expression.
 *  Represents a ternary operation, (cond) ? truevalue : falsevalue.
 *  @extends if_data
 */
class if2t : public if_expr_methods
{
public:
  /** Primary constructor
   *  @param type Type this expression evaluates to.
   *  @param cond Condition to evaulate which side of ternary operator is used.
   *  @param trueval Value to use if cond evaluates to true.
   *  @param falseval Value to use if cond evaluates to false.
   */
  if2t(
    const type2tc &type,
    const expr2tc &cond,
    const expr2tc &trueval,
    const expr2tc &falseval)
    : if_expr_methods(type, if_id, cond, trueval, falseval)
  {
    assert(type->type_id == trueval->type->type_id);
    assert(type->type_id == falseval->type->type_id);
  }
  if2t(const if2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Defines one of the six binary boolean relation nodes. Each takes two
 *  operands of any matching scalar/pointer type, has boolean result type,
 *  routes through its generated _expr_methods base for cmp/lt/crc/hash,
 *  and provides an out-of-line do_simplify override in
 *  src/util/expr_simplifier.cpp. The concrete class names and the
 *  generated is_<name>2t / to_<name>2t helpers are preserved verbatim.
 *  @extends relation_data */
#define ESBMC_DEFINE_RELATION2T(name)                                          \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const expr2tc &v1, const expr2tc &v2)                             \
      : name##_expr_methods(get_bool_type(), name##_id, v1, v2)                \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_RELATION2T(equality);
ESBMC_DEFINE_RELATION2T(notequal);
ESBMC_DEFINE_RELATION2T(lessthan);
ESBMC_DEFINE_RELATION2T(greaterthan);
ESBMC_DEFINE_RELATION2T(lessthanequal);
ESBMC_DEFINE_RELATION2T(greaterthanequal);
#undef ESBMC_DEFINE_RELATION2T

/* The macros below fold sets of `*2t` classes that share both a base
 * data class AND a constructor shape. Grouping is strict on those two
 * properties so any constructor-time invariant we add later applies
 * uniformly to every class in a macro's family. */

/** Arithmetic two-operand node (`add`/`sub`/`mul`/`div`/`modulus`).
 *  @extends arith_2ops */
#define ESBMC_DEFINE_ARITH_2OP(name)                                           \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)        \
      : name##_expr_methods(type, name##_id, v1, v2)                           \
    {                                                                          \
      assert_arith_2ops_consistency(type, name##_id, v1, v2);                  \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_ARITH_2OP(add);
ESBMC_DEFINE_ARITH_2OP(sub);
ESBMC_DEFINE_ARITH_2OP(mul);
ESBMC_DEFINE_ARITH_2OP(div);
ESBMC_DEFINE_ARITH_2OP(modulus);
#undef ESBMC_DEFINE_ARITH_2OP

/** Bitwise / shift two-operand node (`bitand`/`bitor`/`bitxor`/
 *  `shl`/`ashr`/`lshr`). @extends bit_2ops */
#define ESBMC_DEFINE_BIT_2OP(name)                                             \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const type2tc &type, const expr2tc &v1, const expr2tc &v2)        \
      : name##_expr_methods(type, name##_id, v1, v2)                           \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_BIT_2OP(bitand);
ESBMC_DEFINE_BIT_2OP(bitor);
ESBMC_DEFINE_BIT_2OP(bitxor);
ESBMC_DEFINE_BIT_2OP(lshr);
ESBMC_DEFINE_BIT_2OP(shl);
ESBMC_DEFINE_BIT_2OP(ashr);
#undef ESBMC_DEFINE_BIT_2OP

/** Arithmetic one-operand node (`neg`/`abs`/`bswap`/`bitnot`).
 *  @extends arith_1op */
#define ESBMC_DEFINE_ARITH_1OP(name)                                           \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const type2tc &type, const expr2tc &v)                            \
      : name##_expr_methods(type, name##_id, v)                                \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_ARITH_1OP(neg);
ESBMC_DEFINE_ARITH_1OP(abs);
ESBMC_DEFINE_ARITH_1OP(bitnot);
ESBMC_DEFINE_ARITH_1OP(bswap);
#undef ESBMC_DEFINE_ARITH_1OP

/** Pointer one-operand node (`pointer_object`/`pointer_capability`).
 *  @extends pointer_ops */
#define ESBMC_DEFINE_POINTER_1OP(name)                                         \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const type2tc &type, const expr2tc &v)                            \
      : name##_expr_methods(type, name##_id, v)                                \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_POINTER_1OP(pointer_object);
ESBMC_DEFINE_POINTER_1OP(pointer_capability);
#undef ESBMC_DEFINE_POINTER_1OP

/** Logical two-operand boolean-result node. Used for `and`/`or`/`xor`/
 *  `implies` and the Python runtime predicates `isinstance`/`hasattr`/
 *  `isnone`. Implicit `get_bool_type()` result. @extends logic_2ops */
#define ESBMC_DEFINE_LOGIC_2OP(name)                                           \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const expr2tc &s1, const expr2tc &s2)                             \
      : name##_expr_methods(get_bool_type(), name##_id, s1, s2)                \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
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
 *  `isnormal`/`isfinite`). Implicit `get_bool_type()` result.
 *  @extends bool_1op */
#define ESBMC_DEFINE_FP_PREDICATE_1OP(name)                                    \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const expr2tc &operand)                                           \
      : name##_expr_methods(get_bool_type(), name##_id, operand)               \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_FP_PREDICATE_1OP(isnan);
ESBMC_DEFINE_FP_PREDICATE_1OP(isinf);
ESBMC_DEFINE_FP_PREDICATE_1OP(isnormal);
ESBMC_DEFINE_FP_PREDICATE_1OP(isfinite);
#undef ESBMC_DEFINE_FP_PREDICATE_1OP

/** Pointer-object boolean predicate (`valid_object`/`races_check`/
 *  `deallocated_obj`). Implicit `get_bool_type()` result.
 *  @extends object_ops */
#define ESBMC_DEFINE_OBJECT_PREDICATE_1OP(name)                                \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const expr2tc &operand)                                           \
      : name##_expr_methods(get_bool_type(), name##_id, operand)               \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_OBJECT_PREDICATE_1OP(valid_object);
ESBMC_DEFINE_OBJECT_PREDICATE_1OP(races_check);
ESBMC_DEFINE_OBJECT_PREDICATE_1OP(deallocated_obj);
#undef ESBMC_DEFINE_OBJECT_PREDICATE_1OP

/** Pointer-object size-returning op (`capability_base`/`capability_top`/
 *  `dynamic_size`). Implicit `size_type2()` result. @extends object_ops */
#define ESBMC_DEFINE_OBJECT_SIZE_1OP(name)                                     \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const expr2tc &operand)                                           \
      : name##_expr_methods(size_type2(), name##_id, operand)                  \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_OBJECT_SIZE_1OP(dynamic_size);
ESBMC_DEFINE_OBJECT_SIZE_1OP(capability_base);
ESBMC_DEFINE_OBJECT_SIZE_1OP(capability_top);
#undef ESBMC_DEFINE_OBJECT_SIZE_1OP

/** Single-operand overflow-family op returning int32 (`signbit`/
 *  `popcount`). @extends overflow_ops */
#define ESBMC_DEFINE_OVERFLOW_INT32_1OP(name)                                  \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const expr2tc &operand)                                           \
      : name##_expr_methods(get_int32_type(), name##_id, operand)              \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_OVERFLOW_INT32_1OP(signbit);
ESBMC_DEFINE_OVERFLOW_INT32_1OP(popcount);
#undef ESBMC_DEFINE_OVERFLOW_INT32_1OP

/** Marker node holding only a `type` (no operands), derived from
 *  expr2t directly. Used for `unknown`/`invalid`/`null_object`.
 *  @extends expr2t */
#define ESBMC_DEFINE_TYPE_ONLY(name)                                           \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const type2tc &type) : name##_expr_methods(type, name##_id)       \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_TYPE_ONLY(unknown);
ESBMC_DEFINE_TYPE_ONLY(invalid);
ESBMC_DEFINE_TYPE_ONLY(null_object);
#undef ESBMC_DEFINE_TYPE_ONLY

/** `code_*` statement with empty type and a single `expr2tc` operand
 *  (`code_expression`/`code_return`/`code_free`/`code_cpp_del_array`/
 *  `code_cpp_delete`). @extends code_expression_data */
#define ESBMC_DEFINE_CODE_EXPRESSION_1OP(name)                                 \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const expr2tc &operand)                                           \
      : name##_expr_methods(get_empty_type(), name##_id, operand)              \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_CODE_EXPRESSION_1OP(code_expression);
ESBMC_DEFINE_CODE_EXPRESSION_1OP(code_return);
ESBMC_DEFINE_CODE_EXPRESSION_1OP(code_free);
ESBMC_DEFINE_CODE_EXPRESSION_1OP(code_cpp_del_array);
ESBMC_DEFINE_CODE_EXPRESSION_1OP(code_cpp_delete);
#undef ESBMC_DEFINE_CODE_EXPRESSION_1OP

/** `code_*` declaration carrying `(type, irep_idt name)`. Used for
 *  `code_decl`/`code_dead`. @extends code_decl_data */
#define ESBMC_DEFINE_CODE_DECL(name)                                           \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const type2tc &type, const irep_idt &n)                           \
      : name##_expr_methods(type, name##_id, n)                                \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_CODE_DECL(code_decl);
ESBMC_DEFINE_CODE_DECL(code_dead);
#undef ESBMC_DEFINE_CODE_DECL

/** `code_*` C++ throw-decl carrying a single `std::vector<irep_idt>`
 *  of exception names. Used for `code_cpp_throw_decl`/
 *  `code_cpp_throw_decl_end`. @extends code_cpp_throw_decl_data */
#define ESBMC_DEFINE_CODE_CPP_THROW_DECL(name)                                 \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(const std::vector<irep_idt> &names)                               \
      : name##_expr_methods(get_empty_type(), name##_id, names)                \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_CODE_CPP_THROW_DECL(code_cpp_throw_decl);
ESBMC_DEFINE_CODE_CPP_THROW_DECL(code_cpp_throw_decl_end);
#undef ESBMC_DEFINE_CODE_CPP_THROW_DECL

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
 * @extends relation_data */
class cmp_three_way2t : public cmp_three_way_expr_methods
{
public:
  cmp_three_way2t(const type2tc &t, const expr2tc &v1, const expr2tc &v2)
    : cmp_three_way_expr_methods(t, cmp_three_way_id, v1, v2)
  {
  }
  cmp_three_way2t(const cmp_three_way2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Not operation. Inverts boolean operand. Always has boolean type.
 *  @extends bool_1op */
class not2t : public not_expr_methods
{
public:
  /** Primary constructor. @param val Boolean typed operand to invert. */
  not2t(const expr2tc &val) : not_expr_methods(get_bool_type(), not_id, val)
  {
  }
  not2t(const not2t &ref) = default;

  expr2tc do_simplify() const override;

  // Flat-layout migration marker (issue #4560): user fields only, no
  // expr_id/type slots (this is a notype kind — type is always bool).
  static constexpr auto fields = std::make_tuple(&not2t::value);
  static std::string field_names[esbmct::num_type_fields];
};

/** Defines an IEEE two-operand floating-point arithmetic node
 *  (add/sub/mul/div). Each takes two operands and a rounding mode,
 *  has matching floatbv operand/result types, routes through its
 *  generated _expr_methods base for cmp/lt/crc/hash, and provides an
 *  out-of-line do_simplify override in src/util/expr_simplifier.cpp.
 *  The concrete class names and the generated is_<name>2t /
 *  to_<name>2t helpers are preserved verbatim.
 *  @extends ieee_arith_2ops */
#define ESBMC_DEFINE_IEEE_ARITH_2OP(name)                                      \
  class name##2t : public name##_expr_methods                                  \
  {                                                                            \
  public:                                                                      \
    name##2t(                                                                  \
      const type2tc &type,                                                     \
      const expr2tc &v1,                                                       \
      const expr2tc &v2,                                                       \
      const expr2tc &rm)                                                       \
      : name##_expr_methods(type, name##_id, rm, v1, v2)                       \
    {                                                                          \
    }                                                                          \
    name##2t(const name##2t & ref) = default;                                  \
    expr2tc do_simplify() const override;                                      \
    static std::string field_names[esbmct::num_type_fields];                   \
  }

ESBMC_DEFINE_IEEE_ARITH_2OP(ieee_add);
ESBMC_DEFINE_IEEE_ARITH_2OP(ieee_sub);
ESBMC_DEFINE_IEEE_ARITH_2OP(ieee_mul);
ESBMC_DEFINE_IEEE_ARITH_2OP(ieee_div);
#undef ESBMC_DEFINE_IEEE_ARITH_2OP

/** IEEE fused multiply-add operation. Computes (x*y) + z as if to infinite
 *  precision and rounded only once to fit the result type. Must be
 *  floatbvs types. Types of the 3 operands and expr type should match.
 *  @extends ieee_arith_2ops */
class ieee_fma2t : public ieee_fma_expr_methods
{
public:
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
    : ieee_fma_expr_methods(type, ieee_fma_id, rm, v1, v2, v3)
  {
  }
  ieee_fma2t(const ieee_fma2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** IEEE sqrt operation. Square root of the first operand. Must be a
 *  floatbv.
 *  @extends ieee_arith_2ops */
class ieee_sqrt2t : public ieee_sqrt_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of this expr.
   *  @param v1 Operand to take the square root of.
   *  @param rm Rounding mode. */
  ieee_sqrt2t(const type2tc &type, const expr2tc &v1, const expr2tc &rm)
    : ieee_sqrt_expr_methods(type, ieee_sqrt_id, rm, v1)
  {
  }
  ieee_sqrt2t(const ieee_sqrt2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Same-object operation. Checks whether two operands with pointer type have the
 *  same pointer object or not. Always has boolean result.
 *  @extends same_object_data */
class same_object2t : public same_object_expr_methods
{
public:
  /** Primary constructor. @param v1 First object. @param v2 Second object. */
  same_object2t(const expr2tc &v1, const expr2tc &v2)
    : same_object_expr_methods(get_bool_type(), same_object_id, v1, v2)
  {
  }
  same_object2t(const same_object2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Extract pointer offset. From an expression of pointer type, produce the
 *  number of bytes difference between where this pointer points to and the start
 *  of the object it points at. @extends pointer_ops */
class pointer_offset2t : public pointer_offset_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Model basic integer type.
   *  @param ptrobj Pointer object to get offset from. */
  pointer_offset2t(const type2tc &type, const expr2tc &ptrobj)
    : pointer_offset_expr_methods(type, pointer_offset_id, ptrobj)
  {
    assert(type->type_id == type2t::signedbv_id);
    assert(type->get_width() == config.ansi_c.address_width);
  }
  pointer_offset2t(const pointer_offset2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Address of operation. Takes some object as an argument - ideally a symbol
 *  renamed to level 1, unfortunately some string constants reach here. Produces
 *  pointer typed expression.
 *  @extends pointer_ops */
class address_of2t : public address_of_expr_methods
{
public:
  /** Primary constructor.
   *  @param subtype Subtype of pointer to generate. Crucially, the type of the
   *         expr is a pointer to this subtype. This is slightly unintuitive,
   *         might be changed in the future.
   *  @param ptrobj Item to take pointer to. */
  address_of2t(const type2tc &subtype, const expr2tc &ptrobj)
    : address_of_expr_methods(pointer_type2tc(subtype), address_of_id, ptrobj)
  {
    assert(ptrobj->expr_id != expr2t::constant_int_id);
    assert(ptrobj->expr_id != expr2t::address_of_id);
  }
  address_of2t(const address_of2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Extract byte from data. From a particular data structure, extracts a single
 *  byte from its byte representation, at a particular offset into the data
 *  structure. Must only evaluate to byte types.
 *  @extends byte_extract_data */
class byte_extract2t : public byte_extract_expr_methods
{
public:
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
    : byte_extract_expr_methods(
        type,
        byte_extract_id,
        source,
        offset,
        is_big_endian)
  {
  }
  byte_extract2t(const byte_extract2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Update byte. Takes a data object and updates the value of a particular
 *  byte in its byte representation, at a particular offset into the data object.
 *  Output of expression is a new copy of the source object, with the updated
 *  value. @extends byte_update_data */
class byte_update2t : public byte_update_expr_methods
{
public:
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
    : byte_update_expr_methods(
        type,
        byte_update_id,
        source,
        offset,
        updateval,
        is_big_endian)
  {
  }
  byte_update2t(const byte_update2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** With operation. Updates either an array or a struct/union with a new element
 *  or member. Expression value is the array or struct/union with the updated
 *  value. Ideally in the future this will become two operations, one for arrays
 *  and one for structs/unions. @extends with_data */
class with2t : public with_expr_methods
{
  void assert_consistency() const;

public:
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
    : with_expr_methods(type, with_id, source, field, value)
  {
#ifndef NDEBUG /* only check consistency in non-Release builds */
    assert_consistency();
#endif
  }
  with2t(const with2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Member operation. Extracts a particular member out of a struct or union.
 *  @extends member_data */
class member2t : public member_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of extracted member.
   *  @param source Data structure to extract from.
   *  @param memb Name of member to extract.  */
  member2t(const type2tc &type, const expr2tc &source, const irep_idt &memb)
    : member_expr_methods(type, member_id, source, memb)
  {
#ifndef NDEBUG /* only check consistency in non-Release builds */
    assert(
      source->type->type_id == type2t::struct_id ||
      source->type->type_id == type2t::union_id ||
      source->type->type_id == type2t::complex_id);
    auto *data = dynamic_cast<const struct_union_data *>(source->type.get());
    assert(data);
    /* member must exist exactly once in the parent struct/union */
    assert(data->get_component_number(memb).has_value());
#endif
  }
  member2t(const member2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Member reference
 *  @extends member_ref_data */
class member_ref2t : public member_ref_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of extracted member.
   *  @param memb Name of member to extract.  */
  member_ref2t(const type2tc &type, const irep_idt &memb)
    : member_ref_expr_methods(type, member_ref_id, memb)
  {
  }
  member_ref2t(const member_ref2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

/** Member pointer
 *  @extends ptr_mem_data */
class ptr_mem2t : public ptr_mem_expr_methods
{
public:
  /** Primary constructor.
   *  @param source Data structure to extract from.
   *  @param pointer Pointer to member.  */
  ptr_mem2t(const type2tc &type, const expr2tc &source, const expr2tc &pointer)
    : ptr_mem_expr_methods(type, ptr_mem_id, source, pointer)
  {
  }
  ptr_mem2t(const ptr_mem2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

/** Array index operation. Extracts an element from an array at a particular
 *  index. @extends index_data */
class index2t : public index_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of element extracted.
   *  @param source Array to extract data from.
   *  @param index Element in source to extract from. */
  index2t(const type2tc &type, const expr2tc &source, const expr2tc &index)
    : index_expr_methods(type, index_id, source, index)
  {
    assert(is_array_type(source) || is_vector_type(source));
#if 0
    assert(
      is_array_type(source)
        ? type == to_array_type(source->type).subtype
        : ((is_unsignedbv_type(type) || is_signedbv_type(type)) &&
           type->get_width() == config.ansi_c.char_width));
#endif
  }
  index2t(const index2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Check whether operand overflows. Operand must be either add, subtract,
 *  or multiply, and have integer operands themselves. If the result of the
 *  operation doesn't fit in the bitwidth of the operands, this expr evaluates
 *  to true. XXXjmorse - in the future we should ensure the type of the
 *  operand is the expected type result of the operation. That way we can tell
 *  whether to do a signed or unsigned over/underflow test.
 *  @extends overflow_ops */
class overflow2t : public overflow_expr_methods
{
public:
  /** Primary constructor.
   *  @param operand Operation to test overflow on; either an add, subtract, or
   *         multiply. */
  overflow2t(const expr2tc &operand)
    : overflow_expr_methods(get_bool_type(), overflow_id, operand)
  {
  }
  overflow2t(const overflow2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Test if a cast overflows. Check to see whether casting the operand to a
 *  particular bitsize will cause an integer overflow. If it does, this expr
 *  evaluates to true. @extends overflow_cast_data */
class overflow_cast2t : public overflow_cast_expr_methods
{
public:
  /** Primary constructor.
   *  @param operand Value to test cast out on. Should have integer type.
   *  @param bits Number of integer bits to cast operand to.  */
  overflow_cast2t(const expr2tc &operand, unsigned int bits)
    : overflow_cast_expr_methods(
        get_bool_type(),
        overflow_cast_id,
        operand,
        bits)
  {
  }
  overflow_cast2t(const overflow_cast2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

/** Test for negation overflows. Check whether or not negating an operand would
 *  lead to an integer overflow - for example, there's no representation of
 *  -INT_MIN. Evaluates to true if overflow would occur. @extends overflow_ops */
class overflow_neg2t : public overflow_neg_expr_methods
{
public:
  /** Primary constructor. @param operand Integer to test negation of. */
  overflow_neg2t(const expr2tc &operand)
    : overflow_neg_expr_methods(get_bool_type(), overflow_neg_id, operand)
  {
  }
  overflow_neg2t(const overflow_neg2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

/** Record a dynamicly allocated object. Exclusively for use in pointer analysis.
 *  @extends dynamic_object_data */
class dynamic_object2t : public dynamic_object_expr_methods
{
public:
  dynamic_object2t(
    const type2tc &type,
    const expr2tc &inst,
    bool inv,
    bool uknown)
    : dynamic_object_expr_methods(type, dynamic_object_id, inst, inv, uknown)
  {
  }
  dynamic_object2t(const dynamic_object2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

/** Dereference operation. Expanded by symbolic execution into an if-then-else
 *  set of cases that take the value set of what this pointer might point at,
 *  examines the pointer's pointer object, and constructs a huge if-then-else
 *  case to evaluate to the appropriate data object for this pointer.
 *  @extends dereference_data */
class dereference2t : public dereference_expr_methods
{
public:
  /** Primary constructor.
   *  @param type Type of dereferenced data.
   *  @param operand Pointer to dereference. */
  dereference2t(const type2tc &type, const expr2tc &operand)
    : dereference_expr_methods(type, dereference_id, operand)
  {
  }
  dereference2t(const dereference2t &ref) = default;

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
 *  @extends sideeffect_data */
class sideeffect2t : public sideeffect_expr_methods
{
public:
  using allockind = sideeffect_allockind;

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
    : sideeffect_expr_methods(t, sideeffect_id, oper, sz, a, alloct, k)
  {
    if (k == sideeffect_allockind::alloca)
      assert(oper->type == sz->type);
  }
  sideeffect2t(const sideeffect2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class code_block2t : public code_block_expr_methods
{
public:
  code_block2t(const std::vector<expr2tc> &operands)
    : code_block_expr_methods(get_empty_type(), code_block_id, operands)
  {
  }
  code_block2t(const code_block2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class code_assign2t : public code_assign_expr_methods
{
public:
  code_assign2t(const expr2tc &target, const expr2tc &source)
    : code_assign_expr_methods(get_empty_type(), code_assign_id, target, source)
  {
  }
  code_assign2t(const code_assign2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class code_printf2t : public code_printf_expr_methods
{
public:
  code_printf2t(const std::vector<expr2tc> &opers, printf_kindt k)
    : code_printf_expr_methods(get_empty_type(), code_printf_id, opers, k)
  {
  }
  code_printf2t(const code_printf2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class code_skip2t : public code_skip_expr_methods
{
public:
  code_skip2t(const type2tc &type) : code_skip_expr_methods(type, code_skip_id)
  {
  }
  code_skip2t(const code_skip2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class code_goto2t : public code_goto_expr_methods
{
public:
  code_goto2t(const irep_idt &targ)
    : code_goto_expr_methods(get_empty_type(), code_goto_id, targ)
  {
  }
  code_goto2t(const code_goto2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class object_descriptor2t : public object_descriptor_expr_methods
{
public:
  object_descriptor2t(
    const type2tc &t,
    const expr2tc &root,
    const expr2tc &offs,
    unsigned int alignment)
    : object_descriptor_expr_methods(
        t,
        object_descriptor_id,
        root,
        offs,
        alignment)
  {
  }
  object_descriptor2t(const object_descriptor2t &ref) = default;

  const expr2tc &get_root_object() const;

  static std::string field_names[esbmct::num_type_fields];
};

class code_function_call2t : public code_function_call_expr_methods
{
public:
  code_function_call2t(
    const expr2tc &r,
    const expr2tc &func,
    const std::vector<expr2tc> &args)
    : code_function_call_expr_methods(
        get_empty_type(),
        code_function_call_id,
        r,
        func,
        args)
  {
  }
  code_function_call2t(const code_function_call2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class code_comma2t : public code_comma_expr_methods
{
public:
  code_comma2t(const type2tc &t, const expr2tc &s1, const expr2tc &s2)
    : code_comma_expr_methods(t, code_comma_id, s1, s2)
  {
  }
  code_comma2t(const code_comma2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class invalid_pointer2t : public invalid_pointer_expr_methods
{
public:
  invalid_pointer2t(const expr2tc &obj)
    : invalid_pointer_expr_methods(get_bool_type(), invalid_pointer_id, obj)
  {
  }
  invalid_pointer2t(const invalid_pointer2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class code_asm2t : public code_asm_expr_methods
{
public:
  code_asm2t(const type2tc &type, const irep_idt &stringref)
    : code_asm_expr_methods(type, code_asm_id, stringref)
  {
  }
  code_asm2t(const code_asm2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class code_cpp_catch2t : public code_cpp_catch_expr_methods
{
public:
  code_cpp_catch2t(const std::vector<irep_idt> &el)
    : code_cpp_catch_expr_methods(get_empty_type(), code_cpp_catch_id, el)
  {
  }
  code_cpp_catch2t(const code_cpp_catch2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

class code_cpp_throw2t : public code_cpp_throw_expr_methods
{
public:
  code_cpp_throw2t(const expr2tc &o, const std::vector<irep_idt> &l)
    : code_cpp_throw_expr_methods(get_empty_type(), code_cpp_throw_id, o, l)
  {
  }
  code_cpp_throw2t(const code_cpp_throw2t &ref) = default;

  static std::string field_names[esbmct::num_type_fields];
};

/** @extends bit_2ops */
class concat2t : public concat_expr_methods
{
public:
  concat2t(const type2tc &type, const expr2tc &forward, const expr2tc &aft)
    : concat_expr_methods(type, concat_id, forward, aft)
  {
    assert(is_unsignedbv_type(forward));
    assert(is_unsignedbv_type(aft));
  }
  concat2t(const concat2t &ref) = default;

  expr2tc do_simplify() const override;

  static std::string field_names[esbmct::num_type_fields];
};

class extract2t : public extract_expr_methods
{
public:
  extract2t(
    const type2tc &type,
    const expr2tc &from,
    unsigned int upper,
    unsigned int lower)
    : extract_expr_methods(type, extract_id, from, upper, lower)
  {
  }
  extract2t(const extract2t &ref) = default;

  expr2tc do_simplify() const override;

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
