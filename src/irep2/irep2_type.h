#ifndef IREP2_TYPE_H_
#define IREP2_TYPE_H_

#include <optional>
#include <irep2/irep2.h>
#include <util/type.h>

// Forward-declare a concrete <kind>_type2t class for every entry in
// type_kinds.inc. The same manifest drives the type_ids enum in
// irep2.h and the is_/to_/try_to_ predicate generators below.
#define IREP2_TYPE(kind, pretty) class kind##_type2t;
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
// bv_type2t is a shared base of unsignedbv_type2t and signedbv_type2t;
// it isn't itself a kind in the manifest so forward-declare it here.
class bv_type2t;

// We also require in advance, the actual classes that store type data.

class struct_union_data : public type2t
{
public:
  struct_union_data(
    type2t::type_ids id,
    std::vector<type2tc> membs,
    std::vector<irep_idt> names,
    std::vector<irep_idt> pretty_names,
    const irep_idt &n,
    bool _packed)
    : type2t(id),
      members(std::move(membs)),
      member_names(std::move(names)),
      member_pretty_names(std::move(pretty_names)),
      name(n),
      packed(_packed)
  {
  }
  struct_union_data(const struct_union_data &ref) = default;

  /** Fetch index number of member. Looks up the position of @p name in
   *  member_names. Returns std::nullopt when there is no match or more
   *  than one match (the latter is malformed IR); callers that treat
   *  either case as a logic bug should `.value()` the result and let
   *  std::bad_optional_access surface, or assert before unwrapping.
   *  @param name Name of member of this struct/union to look up.
   *  @return Index into members/member_names vectors, or nullopt. */
  std::optional<unsigned int> get_component_number(const irep_idt &name) const;

  const std::vector<type2tc> &get_structure_members() const;
  const std::vector<irep_idt> &get_structure_member_names() const;

  std::vector<type2tc> members;
  std::vector<irep_idt> member_names;
  std::vector<irep_idt> member_pretty_names;
  irep_idt name;
  bool packed;
};

class array_data : public type2t
{
public:
  array_data(type2t::type_ids id, const type2tc &st, const expr2tc &sz, bool i)
    : type2t(id), subtype(st), array_size(sz), size_is_infinite(i)
  {
  }
  array_data(const array_data &ref) = default;

  type2tc subtype;
  expr2tc array_size;
  bool size_is_infinite;
};

// Then give them a typedef name

#define irep_typedefs(basename)                                                \
  template <typename... Args>                                                  \
  inline type2tc basename##_type2tc(Args &&...args)                            \
  {                                                                            \
    return make_irep<basename##_type2t>(std::forward<Args>(args)...);          \
  }

irep_typedefs(bool);
irep_typedefs(empty);
irep_typedefs(symbol);
irep_typedefs(struct);
irep_typedefs(union);
irep_typedefs(unsignedbv);
irep_typedefs(signedbv);
irep_typedefs(code);
irep_typedefs(array);
irep_typedefs(pointer);
irep_typedefs(fixedbv);
irep_typedefs(floatbv);
irep_typedefs(complex);
irep_typedefs(cpp_name);
irep_typedefs(vector);
#undef irep_typedefs

/** Boolean type.
 *  Identifies a boolean type. Contains no additional data.
 *  @extends type2t
 */
class bool_type2t : public type2t
{
public:
  bool_type2t() : type2t(bool_id)
  {
  }
  bool_type2t(const bool_type2t &ref) = default;
  unsigned int get_width() const;

  static constexpr auto fields = std::make_tuple();
  static std::string field_names[esbmct::num_type_fields];
};

/** Empty type.
 *  For void pointers and the like, with no type. No extra data.
 *  @extends type2t
 */
class empty_type2t : public type2t
{
public:
  empty_type2t() : type2t(empty_id)
  {
  }
  empty_type2t(const empty_type2t &ref) = default;
  unsigned int get_width() const;

  static constexpr auto fields = std::make_tuple();
  static std::string field_names[esbmct::num_type_fields];
};

/** Symbolic type.
 *  Temporary, prior to linking up types after parsing, or when a struct/array
 *  contains a recursive pointer to its own type.
 */
class symbol_type2t : public type2t
{
public:
  /** Primary constructor. @param sym_name Name of symbolic type. */
  symbol_type2t(const irep_idt &sym_name)
    : type2t(symbol_id), symbol_name(sym_name)
  {
  }
  symbol_type2t(const symbol_type2t &ref) = default;
  unsigned int get_width() const;

  irep_idt symbol_name;

  static constexpr auto fields =
    std::make_tuple(&symbol_type2t::symbol_name);
  static std::string field_names[esbmct::num_type_fields];
};

/** Struct type.
 *  Represents both C structs and the data in C++ classes. Contains a vector
 *  of types recording what type each member is, a vector of names recording
 *  what the member names are, and a name for the struct.
 *  @extends struct_union_data
 */
class struct_type2t : public struct_union_data
{
public:
  /** Primary constructor.
   *  @param members Vector of types for the members in this struct.
   *  @param memb_names Vector of names for the members in this struct.
   *  @param name Name of this struct.
   */
  struct_type2t(
    const std::vector<type2tc> &members,
    const std::vector<irep_idt> &memb_names,
    const std::vector<irep_idt> &memb_pretty_names,
    const irep_idt &name,
    bool packed = false)
    : struct_union_data(
        struct_id,
        members,
        memb_names,
        memb_pretty_names,
        name,
        packed)
  {
  }
  struct_type2t(const struct_type2t &ref) = default;
  unsigned int get_width() const;

  static constexpr auto fields = std::make_tuple(
    &struct_union_data::members,
    &struct_union_data::member_names,
    &struct_union_data::member_pretty_names,
    &struct_union_data::name,
    &struct_union_data::packed);
  static std::string field_names[esbmct::num_type_fields];
};

/** Union type.
 *  Represents a union type - in a similar vein to struct_type2t, this contains
 *  a vector of types and vector of names, each element of which corresponds to
 *  a member in the union. There's also a name for the union.
 *  @extends struct_union_data
 */
class union_type2t : public struct_union_data
{
public:
  /** Primary constructor.
   *  @param members Vector of types corresponding to each member of union.
   *  @param memb_names Vector of names corresponding to each member of union.
   *  @param name Name of this union
   */
  union_type2t(
    const std::vector<type2tc> &members,
    const std::vector<irep_idt> &memb_names,
    const std::vector<irep_idt> &memb_pretty_names,
    const irep_idt &name,
    bool packed = false)
    : struct_union_data(
        union_id,
        members,
        memb_names,
        memb_pretty_names,
        name,
        packed)
  {
  }
  union_type2t(const union_type2t &ref) = default;
  unsigned int get_width() const;

  static constexpr auto fields = std::make_tuple(
    &struct_union_data::members,
    &struct_union_data::member_names,
    &struct_union_data::member_pretty_names,
    &struct_union_data::name,
    &struct_union_data::packed);
  static std::string field_names[esbmct::num_type_fields];
};

/** Unsigned integer type.
 *  Represents any form of unsigned integer; the size of this integer is
 *  recorded in the width field.
 */
class unsignedbv_type2t : public type2t
{
public:
  /** Primary constructor. @param width Width of represented integer */
  unsignedbv_type2t(unsigned int w) : type2t(unsignedbv_id), width(w)
  {
    // assert(w != 0 && "Must have nonzero width for integer type");
    // XXX -- zero sized bitfields are permissible. Oh my.
  }
  unsignedbv_type2t(const unsignedbv_type2t &ref) = default;
  unsigned int get_width() const;

  unsigned int width;

  static constexpr auto fields = std::make_tuple(&unsignedbv_type2t::width);
  static std::string field_names[esbmct::num_type_fields];
};

/** Signed integer type.
 *  Represents any form of signed integer; the size of this integer is
 *  recorded in the width field.
 */
class signedbv_type2t : public type2t
{
public:
  /** Primary constructor. @param width Width of represented integer */
  signedbv_type2t(signed int w) : type2t(signedbv_id), width(w)
  {
  }
  signedbv_type2t(const signedbv_type2t &ref) = default;
  unsigned int get_width() const;

  unsigned int width;

  static constexpr auto fields = std::make_tuple(&signedbv_type2t::width);
  static std::string field_names[esbmct::num_type_fields];
};

/** Type of functions. */
class code_type2t : public type2t
{
public:
  code_type2t(
    const std::vector<type2tc> &args,
    const type2tc &ret,
    const std::vector<irep_idt> &names,
    bool e)
    : type2t(code_id),
      arguments(args),
      ret_type(ret),
      argument_names(names),
      ellipsis(e)
  {
    assert(args.size() == names.size());
  }
  code_type2t(const code_type2t &ref) = default;
  unsigned int get_width() const;

  std::vector<type2tc> arguments;
  type2tc ret_type;
  std::vector<irep_idt> argument_names;
  bool ellipsis;

  static constexpr auto fields = std::make_tuple(
    &code_type2t::arguments,
    &code_type2t::ret_type,
    &code_type2t::argument_names,
    &code_type2t::ellipsis);
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
class array_type2t : public array_data
{
public:
  /** Primary constructor.
   *  @param subtype Type of elements in this array.
   *  @param size Size of this array.
   *  @param inf Whether or not this array is infinitely sized
   */
  array_type2t(const type2tc &_subtype, const expr2tc &size, bool inf)
    : array_data(array_id, _subtype, size, inf)
  {
    // Constant-fold the size expression so identical array types compare
    // equal regardless of how their size was constructed. Skip the work
    // when the size is already a constant_int (the common case from the
    // frontend) — simplify() would just return nil for it but the walk
    // still costs a few cycles. Long-term fix is to normalise sizes at
    // the frontend / migration boundary instead.
    if (!is_nil_expr(size))
    {
      assert(
        size->type->type_id == signedbv_id ||
        size->type->type_id == unsignedbv_id);
      if (size->expr_id != expr2t::constant_int_id)
      {
        expr2tc sz = size->simplify();
        if (!is_nil_expr(sz))
          array_size = sz;
      }
    }
  }
  array_type2t(const array_type2t &ref) = default;

  virtual ~array_type2t() = default;

  unsigned int get_width() const;

  /** Common base for the two array-sizing exceptions thrown by
   *  array_type2t::get_width (and friends). Lets callers `catch (const
   *  array_size_excp &)` for unified handling, or one of the concrete
   *  subclasses below when they need to distinguish infinite from
   *  dynamic. Derives from std::exception so generic exception
   *  machinery (e.g. catch(std::exception &)) sees them too. */
  class array_size_excp : public std::exception
  {
  public:
    const char *what() const noexcept override
    {
      return "array size is not statically known";
    }
  };

  /** Exception for invalid manipulations of an infinitely sized array.
   *  No payload — the array carries no concrete size. */
  class inf_sized_array_excp : public array_size_excp
  {
  public:
    const char *what() const noexcept override
    {
      return "infinite sized array encountered";
    }
  };

  /** Exception for invalid manipulations of dynamically sized arrays.
   *  Stores the symbolic size of the array so the catcher has it
   *  immediately to hand. */
  class dyn_sized_array_excp : public array_size_excp
  {
  public:
    dyn_sized_array_excp(const expr2tc &_size) : size(_size)
    {
    }

    const char *what() const noexcept override
    {
      return "Sizeof nondeterministically sized array encountered";
    }

    expr2tc size;
  };

  static constexpr auto fields = std::make_tuple(
    &array_data::subtype,
    &array_data::array_size,
    &array_data::size_is_infinite);
  static std::string field_names[esbmct::num_type_fields];
};

/** Vector type.
 *  @extends array_data
 */
class vector_type2t : public array_data
{
public:
  /** Primary constructor.
   *  @param subtype Type of elements in this array.
   *  @param size Size of this array.
   *  @param inf Whether or not this array is infinitely sized
   */
  vector_type2t(const type2tc &_subtype, const expr2tc &size)
    : array_data(vector_id, _subtype, size, false)
  {
    // Mirror array_type2t: skip simplify() when the size is already a
    // literal. See note in array_type2t for the normalisation rationale.
    if (!is_nil_expr(size) && size->expr_id != expr2t::constant_int_id)
    {
      expr2tc sz = size->simplify();
      if (!is_nil_expr(sz))
        array_size = sz;
    }
  }
  vector_type2t(const vector_type2t &ref) = default;
  unsigned int get_width() const;
  static constexpr auto fields = std::make_tuple(
    &array_data::subtype,
    &array_data::array_size,
    &array_data::size_is_infinite);
  static std::string field_names[esbmct::num_type_fields];
};

/** Pointer type.
 *  Simply has a subtype, of what it points to. No other attributes.
 */
class pointer_type2t : public type2t
{
public:
  /** Primary constructor. @param subtype Subtype of this pointer */
  pointer_type2t(const type2tc &st, const bool &p = false)
    : type2t(pointer_id), subtype(st), carry_provenance(p)
  {
  }
  pointer_type2t(const pointer_type2t &ref) = default;
  unsigned int get_width() const;

  type2tc subtype;
  bool carry_provenance;

  static constexpr auto fields = std::make_tuple(
    &pointer_type2t::subtype,
    &pointer_type2t::carry_provenance);
  static std::string field_names[esbmct::num_type_fields];
};

/** Fixed bitvector type.
 *  Contains a spec for a fixed bitwidth number -- this is the equivalent of a
 *  fixedbv_spect in the old irep situation. Stores how bits are distributed
 *  over integer bits and fraction bits.
 */
class fixedbv_type2t : public type2t
{
public:
  /** Primary constructor.
   *  @param width Total number of bits in this type of fixedbv
   *  @param integer Number of integer bits in this type of fixedbv
   */
  fixedbv_type2t(unsigned int w, unsigned int ib)
    : type2t(fixedbv_id), width(w), integer_bits(ib)
  {
  }
  fixedbv_type2t(const fixedbv_type2t &ref) = default;
  unsigned int get_width() const;

  unsigned int width;
  unsigned int integer_bits;

  static constexpr auto fields = std::make_tuple(
    &fixedbv_type2t::width,
    &fixedbv_type2t::integer_bits);
  static std::string field_names[esbmct::num_type_fields];
};

/** Floating-point bitvector type.
 *  Contains a spec for a floating point number -- this is the equivalent of a
 *  ieee_float_spect in the old irep situation. Stores how bits are distributed
 *  over fraction bits and exponent bits.
 */
class floatbv_type2t : public type2t
{
public:
  /** Primary constructor.
   *  @param fraction Number of fraction bits in this type of floatbv
   *  @param exponent Number of exponent bits in this type of floatbv
   */
  floatbv_type2t(unsigned int f, unsigned int e)
    : type2t(floatbv_id), fraction(f), exponent(e)
  {
  }
  floatbv_type2t(const floatbv_type2t &ref) = default;
  unsigned int get_width() const;

  unsigned int fraction;
  unsigned int exponent;

  static constexpr auto fields = std::make_tuple(
    &floatbv_type2t::fraction,
    &floatbv_type2t::exponent);
  static std::string field_names[esbmct::num_type_fields];
};

/** Complex number type.
 *  Represents C's _Complex type, carrying the base element type.
 *  @extend complex_data
 */
class complex_type2t : public struct_union_data
{
public:
  complex_type2t(
    const std::vector<type2tc> &members,
    const std::vector<irep_idt> &memb_names,
    const std::vector<irep_idt> &memb_pretty_names,
    const irep_idt &name,
    bool packed = false)
    : struct_union_data(
        complex_id,
        members,
        memb_names,
        memb_pretty_names,
        name,
        packed)
  {
  }
  complex_type2t(const complex_type2t &ref) = default;
  unsigned int get_width() const;

  static constexpr auto fields = std::make_tuple(
    &struct_union_data::members,
    &struct_union_data::member_names,
    &struct_union_data::member_pretty_names,
    &struct_union_data::name,
    &struct_union_data::packed);
  static std::string field_names[esbmct::num_type_fields];
};

/** C++ Name type.
 *  Contains a type name, but also a vector of template parameters.
 *  Something in the C++ frontend uses this; it's precise purpose is unclear.
 */
class cpp_name_type2t : public type2t
{
public:
  /** Primary constructor.
   *  @param n Name of this type.
   *  @param ta Vector of template arguments (types).
   */
  cpp_name_type2t(const irep_idt &n, const std::vector<type2tc> &ta)
    : type2t(cpp_name_id), name(n), template_args(ta)
  {
  }
  cpp_name_type2t(const cpp_name_type2t &ref) = default;

  unsigned int get_width() const;

  irep_idt name;
  std::vector<type2tc> template_args;

  static constexpr auto fields = std::make_tuple(
    &cpp_name_type2t::name,
    &cpp_name_type2t::template_args);
  static std::string field_names[esbmct::num_type_fields];
};

// Generate "is_<name>_type" predicates and "to_<name>_type" downcasts. The
// downcasts route through irep2_checked_type_cast so a bad to_*_type throws
// irep2_cast_error in every build mode rather than invoking undefined
// behaviour under NDEBUG (the previous design redefined dynamic_cast as
// static_cast in release).
#define type_macros(name)                                                      \
  inline bool is_##name##_type(const expr2tc &e)                               \
  {                                                                            \
    return e->type->type_id == type2t::name##_id;                              \
  }                                                                            \
  inline bool is_##name##_type(const type2tc &t)                               \
  {                                                                            \
    return t->type_id == type2t::name##_id;                                    \
  }                                                                            \
  inline const name##_type2t &to_##name##_type(const type2tc &t)               \
  {                                                                            \
    return irep2_checked_type_cast<const name##_type2t>(                       \
      *t.get(), type2t::name##_id, #name);                                     \
  }                                                                            \
  inline name##_type2t &to_##name##_type(type2tc &t)                           \
  {                                                                            \
    return irep2_checked_type_cast<name##_type2t>(                             \
      *t.get(), type2t::name##_id, #name);                                     \
  }                                                                            \
  inline const name##_type2t *try_to_##name##_type(const type2tc &t)           \
  {                                                                            \
    return is_##name##_type(t) ? &to_##name##_type(t) : nullptr;               \
  }

// Instantiate the is_/to_/try_to_ predicate triple for every kind in
// type_kinds.inc. Same manifest as the enum and forward declarations.
#define IREP2_TYPE(kind, pretty) type_macros(kind);
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
#undef type_macros

#endif /* IREP2_TYPE_H_ */
