#ifndef IREP2_TYPE_H_
#define IREP2_TYPE_H_

#include <optional>
#include <irep2/irep2.h>

// Forward-declare a concrete <kind>_type2t class for every entry in
// type_kinds.inc. The same manifest drives the type_ids enum in
// irep2.h and the is_/to_/try_to_ predicate generators below.
#define IREP2_TYPE(kind, pretty) class kind##_type2t;
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE

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

  static constexpr auto fields = std::make_tuple(&symbol_type2t::symbol_name);
  static std::string field_names[esbmct::num_type_fields];
};

/** Struct type.
 *  Represents both C structs and the data in C++ classes. Contains a vector
 *  of types recording what type each member is, a vector of names recording
 *  what the member names are, and a name for the struct.
 */
class struct_type2t : public type2t
{
public:
  /** Primary constructor.
   *  @param members Vector of types for the members in this struct.
   *  @param memb_names Vector of names for the members in this struct.
   *  @param name Name of this struct.
   */
  struct_type2t(
    const std::vector<type2tc> &_members,
    const std::vector<irep_idt> &memb_names,
    const std::vector<irep_idt> &memb_pretty_names,
    const irep_idt &_name,
    bool _packed = false,
    const std::vector<irep_idt> &memb_base_names = {},
    const BigInt &_alignment = 0,
    const irep_idt &_python_aggregate = irep_idt())
    : type2t(struct_id),
      members(_members),
      member_names(memb_names),
      member_pretty_names(memb_pretty_names),
      member_base_names(memb_base_names),
      name(_name),
      packed(_packed),
      alignment(_alignment),
      python_aggregate(_python_aggregate)
  {
    assert(
      memb_base_names.empty() || memb_base_names.size() == _members.size());
  }
  struct_type2t(const struct_type2t &ref) = default;
  unsigned int get_width() const;

  std::vector<type2tc> members;
  std::vector<irep_idt> member_names;
  std::vector<irep_idt> member_pretty_names;
  /// The components' plain `base_name`s -- a different field from the
  /// `#base_name` that code_type2t::argument_base_names carries. Unreflected: a
  /// member's spelling is no part of the struct's identity, so two otherwise
  /// identical structs must still compare equal
  /// (docs/roadmap/frontends-to-irep2.md §46).
  std::vector<irep_idt> member_base_names;
  irep_idt name;
  bool packed;

  /// An explicit `alignas`, in bytes; zero when the record has none. IREP2 does
  /// not otherwise represent it, and add_padding reads it to decide a record's
  /// trailing padding -- an over-aligned empty struct occupies its alignment,
  /// so without it the back-migrated type gets no pad member and a literal of
  /// it stays shorter than its own type (§7.4). Not reflected: two records that
  /// differ only here would otherwise stop comparing equal, which is a wider
  /// change than this repair.
  BigInt alignment;

  /// The Python model-aggregate kind ("tuple", "dict", "optional") that
  /// `#python_aggregate` records; empty for any other struct. Unreflected, like
  /// `alignment`: the tag already names the type. Carried because `in` and
  /// membership dispatch read it, and a user class may share a tuple's tag
  /// prefix (docs/roadmap/scope-python-irep2.md §10.4).
  irep_idt python_aggregate;

  static constexpr auto fields = std::make_tuple(
    &struct_type2t::members,
    &struct_type2t::member_names,
    &struct_type2t::member_pretty_names,
    &struct_type2t::name,
    &struct_type2t::packed);
  /// Covers the three deliberately unreflected members: `member_base_names` (a
  /// member's spelling is no part of the struct's identity), `alignment` (two
  /// records differing only in `alignas` must still compare equal) and
  /// `python_aggregate`.
  static constexpr std::size_t excluded_field_bytes =
    sizeof(std::vector<irep_idt>) + sizeof(BigInt) + sizeof(irep_idt);
  static std::string field_names[esbmct::num_type_fields];
};

/** Union type.
 *  Represents a union type - in a similar vein to struct_type2t, this contains
 *  a vector of types and vector of names, each element of which corresponds to
 *  a member in the union. There's also a name for the union.
 */
class union_type2t : public type2t
{
public:
  /** Primary constructor.
   *  @param members Vector of types corresponding to each member of union.
   *  @param memb_names Vector of names corresponding to each member of union.
   *  @param name Name of this union
   */
  union_type2t(
    const std::vector<type2tc> &_members,
    const std::vector<irep_idt> &memb_names,
    const std::vector<irep_idt> &memb_pretty_names,
    const irep_idt &_name,
    bool _packed = false)
    : type2t(union_id),
      members(_members),
      member_names(memb_names),
      member_pretty_names(memb_pretty_names),
      name(_name),
      packed(_packed)
  {
  }
  union_type2t(const union_type2t &ref) = default;
  unsigned int get_width() const;

  std::vector<type2tc> members;
  std::vector<irep_idt> member_names;
  std::vector<irep_idt> member_pretty_names;
  irep_idt name;
  bool packed;

  static constexpr auto fields = std::make_tuple(
    &union_type2t::members,
    &union_type2t::member_names,
    &union_type2t::member_pretty_names,
    &union_type2t::name,
    &union_type2t::packed);
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
  unsignedbv_type2t(
    unsigned int w,
    bool qualified = false,
    const irep_idt &cpp = irep_idt())
    : type2t(unsignedbv_id),
      width(w),
      cpp_type(cpp),
      constant_qualified(qualified)
  {
    // assert(w != 0 && "Must have nonzero width for integer type");
    // XXX -- zero sized bitfields are permissible. Oh my.
  }
  unsignedbv_type2t(const unsignedbv_type2t &ref) = default;
  unsigned int get_width() const;

  unsigned int width;
  /// The source language's own spelling of this type, as `#cpp_type` records
  /// it. Unreflected: a spelling is no part of the type's identity, so two
  /// bitvectors of the same width are the same type however they were spelled.
  /// Carried because a consumer reads it back -- `python_converter::
  /// get_python_type_category` distinguishes a 1-char string element from an
  /// 8-bit int by it (docs/roadmap/scope-python-irep2.md §9).
  irep_idt cpp_type;
  /// Whether the source qualified this `const`, as `#constant` records it.
  /// Unreflected: `c_expr2string` prints it and nothing else reads it, so two
  /// integers of the same width are the same type whether or not one was
  /// qualified (docs/roadmap/scope-clang-c-irep2.md §158).
  bool constant_qualified;

  static constexpr auto fields = std::make_tuple(&unsignedbv_type2t::width);
  static constexpr std::size_t excluded_field_bytes =
    sizeof(irep_idt) + sizeof(bool);
  /// `cpp_type` pushes `constant_qualified` into a fresh slot, so this class
  /// now carries four bytes of trailing padding and `fields_cover_class` has no
  /// margin left: a further unreflected field would fit in the hole unnoticed.
  /// Pin the size so the next one has to come here first.
  static_assert(
    sizeof(unsigned int) + sizeof(irep_idt) + sizeof(bool) <=
      2 * sizeof(void *),
    "unsignedbv_type2t's reflected + excluded fields no longer fit the pinned "
    "layout; re-check fields_cover_class's margin before adding a field");
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
  signedbv_type2t(
    signed int w,
    bool qualified = false,
    const irep_idt &cpp = irep_idt())
    : type2t(signedbv_id),
      width(w),
      cpp_type(cpp),
      constant_qualified(qualified)
  {
  }
  signedbv_type2t(const signedbv_type2t &ref) = default;
  unsigned int get_width() const;

  unsigned int width;
  /// The source language's own spelling of this type, as `#cpp_type` records
  /// it. Unreflected, for the reason given on unsignedbv_type2t.
  irep_idt cpp_type;
  /// Whether the source qualified this `const`, as `#constant` records it.
  /// Unreflected: `c_expr2string` prints it and nothing else reads it, so two
  /// integers of the same width are the same type whether or not one was
  /// qualified (docs/roadmap/scope-clang-c-irep2.md §158).
  bool constant_qualified;

  static constexpr auto fields = std::make_tuple(&signedbv_type2t::width);
  static constexpr std::size_t excluded_field_bytes =
    sizeof(irep_idt) + sizeof(bool);
  /// `cpp_type` pushes `constant_qualified` into a fresh slot, so this class
  /// now carries four bytes of trailing padding and `fields_cover_class` has no
  /// margin left: a further unreflected field would fit in the hole unnoticed.
  /// Pin the size so the next one has to come here first.
  static_assert(
    sizeof(unsigned int) + sizeof(irep_idt) + sizeof(bool) <=
      2 * sizeof(void *),
    "signedbv_type2t's reflected + excluded fields no longer fit the pinned "
    "layout; re-check fields_cover_class's margin before adding a field");
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
    bool e,
    const std::vector<irep_idt> &base_names = {},
    const std::vector<expr2tc> &defaults = {},
    const irep_idt &exc_kind = irep_idt(),
    const std::vector<irep_idt> &exc_types = {},
    const irep_idt &ret_marker = irep_idt(),
    bool implicit_union_copy_move = false,
    const std::vector<type2tc> &exc_decl = {})
    : type2t(code_id),
      arguments(args),
      ret_type(ret),
      argument_names(names),
      argument_base_names(base_names),
      argument_defaults(defaults),
      exception_types(exc_types),
      exception_decl(exc_decl),
      exception_kind(exc_kind),
      return_marker(ret_marker),
      ellipsis(e),
      implicit_union_copy_move(implicit_union_copy_move)
  {
    assert(args.size() == names.size());
    assert(base_names.empty() || base_names.size() == args.size());
    assert(defaults.empty() || defaults.size() == args.size());
  }
  code_type2t(const code_type2t &ref) = default;
  unsigned int get_width() const;

  std::vector<type2tc> arguments;
  type2tc ret_type;
  std::vector<irep_idt> argument_names;
  /// The arguments' `#base_name`s, carried across the migrate seam but *not*
  /// reflected: C11 6.7.6.3p15 makes a parameter's spelling no part of the
  /// function type, so two signatures differing only here are the same type and
  /// must hash and compare equal. Kept because a consumer reads it back --
  /// clang_cpp_convert_vft.cpp's thunk argument loop does
  /// (docs/roadmap/frontends-to-irep2.md §44).
  std::vector<irep_idt> argument_base_names;
  /// The arguments' `#default_value`s, null where an argument has none, and
  /// empty when none has one. Unreflected: a default is no part of the
  /// function's type. Carried because Python call lowering fills a missing
  /// argument from it (converter_funcall.cpp, function_call/expr.cpp).
  std::vector<expr2tc> argument_defaults;
  /// A resolved C++ exception specification, as `exception_spec_kind` and
  /// `exception_spec_types` record it (util/lang/exception_specification.h);
  /// empty when there is none. Unreflected, like the fields above: two
  /// signatures differing only here still compare equal. Carried because
  /// goto_convert_functions decodes it from the function symbol's type.
  std::vector<irep_idt> exception_types;
  /// A dynamic specification's declared types before
  /// finalize_exception_specification resolves them to `exception_types`.
  std::vector<type2tc> exception_decl;
  irep_idt exception_kind;
  /// A C++ constructor's or destructor's pseudo return type ("constructor" /
  /// "destructor"), which ret_type models as empty; and whether it is an
  /// implicit copy/move constructor of a union. Unreflected, like the fields
  /// above. Carried because vptr initialisation and the union copy/move
  /// synthesis read them (docs/roadmap/frontends-to-irep2.md §50.2,
  /// scope-clang-cpp-irep2.md §12).
  irep_idt return_marker;
  bool ellipsis;
  bool implicit_union_copy_move;

  static constexpr auto fields = std::make_tuple(
    &code_type2t::arguments,
    &code_type2t::ret_type,
    &code_type2t::argument_names,
    &code_type2t::ellipsis);
  static constexpr std::size_t excluded_field_bytes =
    2 * sizeof(std::vector<irep_idt>) + sizeof(std::vector<expr2tc>) +
    sizeof(std::vector<type2tc>) + 2 * sizeof(irep_idt) + sizeof(bool);
  static std::string field_names[esbmct::num_type_fields];
};

/** Array type.
 *  Comes with a subtype of the array and a size that might be constant, might
 *  be nondeterministic, might be infinite. These facts are recorded in the
 *  array_size and size_is_infinite fields.
 *
 *  If size_is_infinite is true, array_size will be null. If array_size is
 *  not a constant number, then it's a dynamically sized array.
 */
class array_type2t : public type2t
{
public:
  /** Primary constructor.
   *  @param subtype Type of elements in this array.
   *  @param size Size of this array.
   *  @param inf Whether or not this array is infinitely sized
   */
  array_type2t(const type2tc &_subtype, const expr2tc &size, bool inf)
    : type2t(array_id),
      subtype(_subtype),
      array_size(size),
      size_is_infinite(inf)
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

  type2tc subtype;
  expr2tc array_size;
  bool size_is_infinite;

  static constexpr auto fields = std::make_tuple(
    &array_type2t::subtype,
    &array_type2t::array_size,
    &array_type2t::size_is_infinite);
  static std::string field_names[esbmct::num_type_fields];
};

/** Vector type. */
class vector_type2t : public type2t
{
public:
  /** Primary constructor.
   *  @param _subtype Type of elements in this vector.
   *  @param size Number of lanes.
   */
  vector_type2t(const type2tc &_subtype, const expr2tc &size)
    : type2t(vector_id),
      subtype(_subtype),
      array_size(size),
      size_is_infinite(false)
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

  type2tc subtype;
  expr2tc array_size;
  bool size_is_infinite;

  static constexpr auto fields = std::make_tuple(
    &vector_type2t::subtype,
    &vector_type2t::array_size,
    &vector_type2t::size_is_infinite);
  static std::string field_names[esbmct::num_type_fields];
};

/** How the source spelled a pointer. The irept form keeps this in the
 *  `#reference` / `#rvalue_reference` attributes, which have no equivalent
 *  here, so a round trip used to erase it. */
enum class pointer_ref_kindt
{
  NONE,
  LVALUE,
  RVALUE
};

/** Pointer type.
 *  Simply has a subtype, of what it points to, and how the source spelled it.
 */
class pointer_type2t : public type2t
{
public:
  /** Primary constructor. @param subtype Subtype of this pointer */
  pointer_type2t(
    const type2tc &st,
    const bool &p = false,
    pointer_ref_kindt rk = pointer_ref_kindt::NONE)
    : type2t(pointer_id), subtype(st), carry_provenance(p), ref_kind(rk)
  {
  }
  pointer_type2t(const pointer_type2t &ref) = default;
  unsigned int get_width() const;

  type2tc subtype;
  bool carry_provenance;
  pointer_ref_kindt ref_kind;

  static constexpr auto fields = std::make_tuple(
    &pointer_type2t::subtype,
    &pointer_type2t::carry_provenance,
    &pointer_type2t::ref_kind);
  static std::string field_names[esbmct::num_type_fields];
};

/** Fixed bitvector type.
 *  Spec for a fixed bitwidth number — stores how the bits are distributed
 *  between integer bits and fraction bits.
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

  static constexpr auto fields =
    std::make_tuple(&fixedbv_type2t::width, &fixedbv_type2t::integer_bits);
  static std::string field_names[esbmct::num_type_fields];
};

/** Floating-point bitvector type.
 *  Spec for a floating point number — stores how the bits are distributed
 *  between fraction bits and exponent bits.
 */
class floatbv_type2t : public type2t
{
public:
  /** Primary constructor.
   *  @param fraction Number of fraction bits in this type of floatbv
   *  @param exponent Number of exponent bits in this type of floatbv
   */
  floatbv_type2t(
    unsigned int f,
    unsigned int e,
    const irep_idt &cpp = irep_idt())
    : type2t(floatbv_id), fraction(f), exponent(e), cpp_type(cpp)
  {
  }
  floatbv_type2t(const floatbv_type2t &ref) = default;
  unsigned int get_width() const;

  unsigned int fraction;
  unsigned int exponent;
  /// The source language's own spelling of this type, as `#cpp_type` records
  /// it. Unreflected, for the reason given on unsignedbv_type2t.
  irep_idt cpp_type;

  static constexpr auto fields =
    std::make_tuple(&floatbv_type2t::fraction, &floatbv_type2t::exponent);
  static constexpr std::size_t excluded_field_bytes = sizeof(irep_idt);
  static std::string field_names[esbmct::num_type_fields];
};

/** Complex number type. C's `_Complex T` is a pair (real, imag) of two
 *  values of the same scalar type T, so the kind carries only the
 *  element type. The SMT tuple lowering synthesises the 2-element view
 *  at the boundary via struct_union_members / struct_union_member_names. */
class complex_type2t : public type2t
{
public:
  complex_type2t(const type2tc &st) : type2t(complex_id), subtype(st)
  {
  }
  complex_type2t(const complex_type2t &ref) = default;
  unsigned int get_width() const;

  type2tc subtype;

  static constexpr auto fields = std::make_tuple(&complex_type2t::subtype);
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

  static constexpr auto fields =
    std::make_tuple(&cpp_name_type2t::name, &cpp_name_type2t::template_args);
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

// struct_type2t and union_type2t each own a 5-field block
// (members / member_names / member_pretty_names / name / packed);
// complex_type2t stores only its element type but the SMT tuple
// lowering treats it as a (real, imag) 2-tuple of that type. The
// helpers below give callers a uniform "tuple view" of all three
// without forcing them to dispatch by hand: for struct/union they
// copy the kind's stored vectors, for complex they synthesise the
// 2-element view from the subtype. Return-by-value lets the helper
// be uniform; callers bind to `const auto &x = ...` and rely on
// temporary lifetime extension.
//
// Small per-call copies in exchange for a single point of truth per
// field. Direct mutation goes via the mutating overload below (or via
// the concrete kind, e.g. `to_struct_type(t).members`).
std::vector<type2tc> struct_union_members(const type2tc &t);
std::vector<irep_idt> struct_union_member_names(const type2tc &t);
irep_idt struct_union_name(const type2tc &t);
bool struct_union_packed(const type2tc &t);

/** Mutating reference to a struct/union's members vector. Unlike the
 *  read-only overload this does NOT synthesise a view for complex_type
 *  (which has no members vector), so it is restricted to struct_id and
 *  union_id; passing anything else aborts. Detaches @p t first via the
 *  non-const `to_*_type` accessor, so the returned reference is safe to
 *  mutate. */
std::vector<type2tc> &struct_union_members(type2tc &t);

/** Index of @p comp in struct/union/complex member_names, or nullopt
 *  if it is missing or duplicated (the latter is malformed IR). */
std::optional<unsigned int>
struct_union_get_component_number(const type2tc &t, const irep_idt &comp);

#endif /* IREP2_TYPE_H_ */
