// Implicit-conversion admission for the IREP2 overload of c_typecast.
//
// check_c_implicit_typecast has an irept and an expr2tc copy, written
// independently (see the note at the top of c_typecast.cpp). The IREP2 copy
// omitted `floatbv` entirely -- absent as a destination in every source branch
// and absent as a source branch of its own -- so it fell through to its final
// `return true` and rejected every implicit conversion involving a float.
// ESBMC represents a float as floatbv, so that rejected the only float
// representation there is, and c_implicit_typecast became
// a silent no-op for its callers: python_adjust's assignment arm left an
// integer stored into a `double` lvalue
// (docs/roadmap/scope-relational-float-reconciliation.md §18.3). The same
// omission was fixed in get_c_type by esbmc/esbmc#6688.

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <util/lang/c_typecast.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>
#include <util/irep/migrate.h>
#include <util/irep/std_expr.h>
#include <util/arith/arith_tools.h>
#include <irep2/irep2_utils.h>

// migrate_expr resolves symbol types through this thread-local, which only
// language_ui sets in a real run. Restoring it matters: it would otherwise
// outlive the TEST_CASE-local namespacet it points at.
struct migrate_lookupt
{
  explicit migrate_lookupt(const namespacet &ns)
    : saved(migrate_namespace_lookup)
  {
    migrate_namespace_lookup = &ns;
  }
  ~migrate_lookupt()
  {
    migrate_namespace_lookup = saved;
  }
  const namespacet *saved;
};

// Differential harness for the two-operand usual-arithmetic conversion. The
// arithmetic sections are grounded in the standard, not in either copy: C17
// 6.3.1.8 converts both operands to one common type, so each section asserts
// that type first and only then asks the two copies to agree byte-for-byte
// after migration. (The pointer sections pin ESBMC's own reconciliation
// choice -- 6.3.1.8 covers arithmetic types only.) The ranking tables were
// not translations of each other: the irept get_c_type ranked 128-bit
// operands OTHER until this change, escaping promotion entirely; a
// drift-only check cannot say which side is wrong, the common-type
// assertion can.
// Fixed-point operands go through the exprt overload only. The expr2tc copy
// deliberately has no FIXED arm -- its callers are all in the python frontend,
// which has no fixed-point types -- so a FIXED operand falls through to its
// switch default and converts neither side (c_typecast.cpp). Asserting parity
// there would pin a promise the code does not make; this harness checks the
// overload that does handle them, and the divergence is pinned separately.
static void require_legacy_arith_operands(
  const namespacet &ns,
  const exprt &a,
  const exprt &b,
  const typet &expected_a,
  const typet &expected_b)
{
  exprt legacy_a = a, legacy_b = b;
  REQUIRE_FALSE(c_implicit_typecast_arithmetic(legacy_a, legacy_b, ns));
  REQUIRE(legacy_a.type() == expected_a);
  REQUIRE(legacy_b.type() == expected_b);
}

// The general form: each operand has its own expected type, so the pointer
// cases (6.5.6 converts neither) share one harness with the arithmetic ones
// (6.3.1.8 converts both to a common type).
static void require_arith_operands(
  const namespacet &ns,
  const exprt &a,
  const exprt &b,
  const typet &expected_a,
  const typet &expected_b)
{
  migrate_lookupt lookup(ns);

  exprt legacy_a = a, legacy_b = b;
  REQUIRE_FALSE(c_implicit_typecast_arithmetic(legacy_a, legacy_b, ns));
  REQUIRE(legacy_a.type() == expected_a);
  REQUIRE(legacy_b.type() == expected_b);

  expr2tc native_a, native_b;
  migrate_expr(a, native_a);
  migrate_expr(b, native_b);
  REQUIRE_FALSE(c_implicit_typecast_arithmetic(native_a, native_b, ns));
  REQUIRE(native_a->type == migrate_type(expected_a));
  REQUIRE(native_b->type == migrate_type(expected_b));

  expr2tc legacy_a_migrated, legacy_b_migrated;
  migrate_expr(legacy_a, legacy_a_migrated);
  migrate_expr(legacy_b, legacy_b_migrated);
  REQUIRE(legacy_a_migrated == native_a);
  REQUIRE(legacy_b_migrated == native_b);
}

static void require_arith_result(
  const namespacet &ns,
  const exprt &a,
  const exprt &b,
  const typet &common_type)
{
  require_arith_operands(ns, a, b, common_type, common_type);
}

// get_c_type ranks an operand against config.ansi_c, which is zero-initialised
// bar int_128_width. Pin a model in main() rather than at namespace scope:
// `config` lives in another translation unit, so a static initialiser here
// would race its constructor.
int main(int argc, char *argv[])
{
  config.ansi_c.set_data_model(configt::LP64);
  return Catch::Session().run(argc, argv);
}

// check_c_implicit_typecast returns *false* to mean "permitted".
TEST_CASE(
  "check_c_implicit_typecast admits floatbv on both sides",
  "[c_typecast]")
{
  const type2tc i32 = get_int32_type();
  const type2tc dbl = double_type2();
  const type2tc flt = float_type2();

  SECTION("integer to floating point")
  {
    REQUIRE_FALSE(check_c_implicit_typecast(i32, dbl));
  }
  SECTION("Boolean to floating point")
  {
    REQUIRE_FALSE(check_c_implicit_typecast(get_bool_type(), dbl));
  }
  SECTION("floating point to integer")
  {
    REQUIRE_FALSE(check_c_implicit_typecast(dbl, i32));
  }
  SECTION("floating point to Boolean")
  {
    REQUIRE_FALSE(check_c_implicit_typecast(dbl, get_bool_type()));
  }
  SECTION("narrowing between floating-point widths")
  {
    REQUIRE_FALSE(check_c_implicit_typecast(dbl, flt));
  }
}

// The admission is not a blanket one: a conversion the irept overload also
// rejects must stay rejected, or the arms above would be untested by
// construction.
TEST_CASE(
  "check_c_implicit_typecast still rejects a struct source",
  "[c_typecast]")
{
  const type2tc st = struct_type2tc(
    std::vector<type2tc>{get_int32_type()},
    std::vector<irep_idt>{"f"},
    std::vector<irep_idt>{"f"},
    "tag-s");

  REQUIRE(check_c_implicit_typecast(st, double_type2()));
}

// The behaviour the callers depend on: the cast is actually inserted. A
// rejected conversion leaves the expression untouched, which is how an integer
// came to be stored into a `double` lvalue.
TEST_CASE("c_implicit_typecast converts an integer to double", "[c_typecast]")
{
  contextt ctx;
  namespacet ns(ctx);

  expr2tc e = gen_one(get_int32_type());
  REQUIRE_FALSE(c_implicit_typecast(e, double_type2(), ns));
  REQUIRE(is_floatbv_type(e->type));
}

// ---------------------------------------------------------------------------
// Admission parity between the 2-argument check_c_implicit_typecast copies.
//
// The floatbv omission this file's header comment describes was found by a
// user-visible bug, not by a test: nothing compared the two copies' verdicts
// systematically. The matrix below does -- every pair of scalar types a
// frontend produces, plus the pointer and non-scalar shapes, must be admitted
// or rejected identically by the irept copy and the expr2tc copy. Verdicts are
// compared between the copies, not recomputed: the C standard's own ground
// truth is pinned separately in "admission pins from the C standard".
//
// Built lazily: c_types.h constructors read config.ansi_c widths, which
// main() pins to LP64 -- a namespace-scope table would race that.
// ---------------------------------------------------------------------------

static bool legacy_admits(const typet &src, const typet &dest)
{
  // check_c_implicit_typecast returns false to mean "permitted".
  return !check_c_implicit_typecast(src, dest);
}

static bool native_admits(const typet &src, const typet &dest)
{
  return !check_c_implicit_typecast(migrate_type(src), migrate_type(dest));
}

// Built per call rather than cached: the entries are constructed from the
// current configuration, which a test case may have adjusted.
static std::vector<std::pair<std::string, typet>> scalar_types()
{
  return {
    {"bool", bool_type()},
    {"signed char", signed_char_type()},
    {"unsigned char", unsigned_char_type()},
    {"short", signed_short_int_type()},
    {"unsigned short", unsigned_short_int_type()},
    {"int", int_type()},
    {"unsigned int", uint_type()},
    {"long", long_int_type()},
    {"unsigned long", long_uint_type()},
    {"long long", long_long_int_type()},
    {"unsigned long long", long_long_uint_type()},
    {"__int128", int128_type()},
    {"unsigned __int128", uint128_type()},
    {"half", half_float_type()},
    {"float", float_type()},
    {"double", double_type()},
    {"long double", long_double_type()},
  };
}

// TR 18037 fixed-point types, built the way clang_c_convertert does from
// clang's FixedPointSemantics: width, integer bits, and the signedness and
// saturation flags. Floats are always floatbv now, so these are the only
// fixedbv values a program can produce.
static typet fixedbv_type(unsigned width, unsigned integer_bits, bool is_signed)
{
  fixedbv_typet t;
  t.set_width(width);
  t.set_integer_bits(integer_bits);
  if (!is_signed)
    t.set("#esbmc_unsigned", "1");
  return t;
}

// _Fract is s0.15 and _Accum s15.16, the natural narrow/wide pair.
static typet fract_type()
{
  return fixedbv_type(16, 1, true);
}
static typet accum_type()
{
  return fixedbv_type(32, 16, true);
}

TEST_CASE(
  "admission parity: every scalar pair agrees across the two copies",
  "[c_typecast]")
{
  for (const auto &[src_name, src] : scalar_types())
    for (const auto &[dest_name, dest] : scalar_types())
    {
      INFO("src: " + src_name + " dest: " + dest_name);
      REQUIRE(legacy_admits(src, dest) == native_admits(src, dest));
    }
}

// Built per call for the same reason as scalar_types().
static std::vector<std::pair<std::string, typet>> pointer_shape_types()
{
  const typet int_ptr = pointer_typet(int_type());
  return {
    {"int*", int_ptr},
    {"void*", pointer_typet(empty_typet())},
    {"char*", pointer_typet(char_type())},
    {"int**", pointer_typet(int_ptr)},
    {"int", int_type()},
    {"bool", bool_type()},
    {"double", double_type()},
  };
}

TEST_CASE("admission parity: pointer shapes", "[c_typecast]")
{
  for (const auto &[src_name, src] : pointer_shape_types())
    for (const auto &[dest_name, dest] : pointer_shape_types())
    {
      INFO("src: " + src_name + " dest: " + dest_name);
      REQUIRE(legacy_admits(src, dest) == native_admits(src, dest));
    }
}

TEST_CASE("admission pins from the C standard", "[c_typecast]")
{
  // C17 6.3.1: any two arithmetic types convert implicitly in both
  // directions, however lossy. Both copies must say so.
  REQUIRE(legacy_admits(int_type(), double_type()));
  REQUIRE(native_admits(int_type(), double_type()));
  REQUIRE(legacy_admits(double_type(), int_type()));
  REQUIRE(native_admits(double_type(), int_type()));
  REQUIRE(legacy_admits(bool_type(), int_type()));
  REQUIRE(native_admits(bool_type(), int_type()));

  // No implicit conversion joins an aggregate and an arithmetic type.
  struct_typet st;
  st.components().push_back(struct_typet::componentt("f", int_type()));

  REQUIRE_FALSE(legacy_admits(st, int_type()));
  REQUIRE_FALSE(native_admits(st, int_type()));
  REQUIRE_FALSE(legacy_admits(int_type(), st));
  REQUIRE_FALSE(native_admits(int_type(), st));

  // Nor between floating-point and pointer in either direction.
  REQUIRE_FALSE(legacy_admits(double_type(), pointer_typet(int_type())));
  REQUIRE_FALSE(native_admits(double_type(), pointer_typet(int_type())));
  REQUIRE_FALSE(legacy_admits(pointer_typet(int_type()), double_type()));
  REQUIRE_FALSE(native_admits(pointer_typet(int_type()), double_type()));
}

// The namespace-taking overloads route through implicit_typecast_followed,
// whose copies intentionally differ on C++-shaped types (references,
// pointer-to-member, struct-to-base-pointer; scope-coupled-arith-assign-
// conversion.md §20.1). The C-shaped subset must still agree.
TEST_CASE(
  "admission parity: the namespace-taking overloads on C-shaped types",
  "[c_typecast]")
{
  contextt ctx;
  namespacet ns(ctx);

  auto require_ns_parity = [&ns](const typet &src, const typet &dest) {
    const bool legacy_failed = check_c_implicit_typecast(src, dest, ns);
    const bool native_failed =
      check_c_implicit_typecast(migrate_type(src), migrate_type(dest), ns);
    REQUIRE(legacy_failed == native_failed);
  };

  for (const auto &[src_name, src] : scalar_types())
    for (const auto &[dest_name, dest] : scalar_types())
    {
      INFO("src: " + src_name + " dest: " + dest_name);
      require_ns_parity(src, dest);
    }

  // struct-to-pointer is excluded: the irept copy treats it as the C++
  // derived-to-base address-of case (§20.1 item 6), the expr2tc copy rejects.
  const typet int_ptr = pointer_typet(int_type());
  const typet void_ptr = pointer_typet(empty_typet());
  for (const auto &[src_name, src] : scalar_types())
  {
    INFO("src: " + src_name + " dest: int*");
    require_ns_parity(src, int_ptr);
    INFO("src: int* dest: " + src_name);
    require_ns_parity(int_ptr, src);
    INFO("src: " + src_name + " dest: void*");
    require_ns_parity(src, void_ptr);
    INFO("src: void* dest: " + src_name);
    require_ns_parity(void_ptr, src);
  }
}

// The same matrix with the TR 18037 fixed-point types added to the scalar
// table. get_c_type ranks fixedbv beside floatbv by width and both
// check_c_implicit_typecast copies admit it in every scalar branch; the
// matrix pins that the two copies stay in lockstep over fixedbv too.
TEST_CASE(
  "admission parity: the scalar matrix including fixed-point types",
  "[c_typecast]")
{
  contextt ctx;
  namespacet ns(ctx);

  auto require_both_overloads = [&ns](const typet &src, const typet &dest) {
    REQUIRE(legacy_admits(src, dest) == native_admits(src, dest));

    const bool legacy_failed = check_c_implicit_typecast(src, dest, ns);
    const bool native_failed =
      check_c_implicit_typecast(migrate_type(src), migrate_type(dest), ns);
    REQUIRE(legacy_failed == native_failed);
  };

  std::vector<std::pair<std::string, typet>> types = scalar_types();
  types.emplace_back("_Fract", fract_type());
  types.emplace_back("_Accum", accum_type());
  types.emplace_back("unsigned _Fract", fixedbv_type(16, 0, false));
  types.emplace_back("unsigned _Accum", fixedbv_type(32, 16, false));

  for (const auto &[src_name, src] : types)
    for (const auto &[dest_name, dest] : types)
    {
      INFO("src: " + src_name + " dest: " + dest_name);
      require_both_overloads(src, dest);
    }
}

// Differential harness. implicit_typecast_followed also exists twice, and the
// two copies are not translations of each other: the irept one additionally
// handles references, pointer-to-member, incomplete_array sources, qualifier
// warnings and string-constant-to-array, none of which the expr2tc one has
// (docs/roadmap/scope-coupled-arith-assign-conversion.md §20). Those gaps are
// C++-frontend shaped; what follows pins the arithmetic and pointer conversions
// that every frontend performs at an assignment, so a future edit to one copy
// cannot silently drift from the other the way the floatbv omission above did.
static void require_overloads_agree(
  const namespacet &ns,
  const exprt &input,
  const typet &dest)
{
  migrate_lookupt lookup(ns);

  exprt legacy = input;
  const bool legacy_failed = c_implicit_typecast(legacy, dest, ns);

  expr2tc native;
  migrate_expr(input, native);
  const bool native_failed =
    c_implicit_typecast(native, migrate_type(dest), ns);

  REQUIRE(legacy_failed == native_failed);

  expr2tc legacy_migrated;
  migrate_expr(legacy, legacy_migrated);
  REQUIRE(legacy_migrated == native);
}

TEST_CASE(
  "both c_implicit_typecast overloads agree on arithmetic conversions",
  "[c_typecast]")
{
  contextt ctx;
  namespacet ns(ctx);

  SECTION("integer widens to double")
  {
    require_overloads_agree(ns, from_integer(1, int_type()), double_type());
  }

  SECTION("integer widens to a wider integer")
  {
    require_overloads_agree(ns, from_integer(1, int_type()), long_int_type());
  }

  SECTION("integer narrows to bool")
  {
    require_overloads_agree(ns, from_integer(1, int_type()), bool_type());
  }

  // A non-constant source takes the unfolded route, so it covers the other
  // side of the branch the constant sections above exercise.
  SECTION("double narrows to integer")
  {
    require_overloads_agree(ns, symbol_exprt("d", double_type()), int_type());
  }

  SECTION("signed converts to unsigned of the same width")
  {
    require_overloads_agree(ns, from_integer(1, int_type()), uint_type());
  }
}

// Wider expr-level differential coverage: every conversion shape a frontend
// performs at an assignment, over both constant (folded) and symbol
// (unfolded) sources. The fold path is #6873's habitat: the copies folded
// differently until that fix, and only a byte-equality comparison after
// migration sees it.
TEST_CASE(
  "both c_implicit_typecast overloads agree on the fold and float paths",
  "[c_typecast]")
{
  contextt ctx;
  namespacet ns(ctx);

  SECTION("Boolean converts to float (symbol)")
  {
    require_overloads_agree(ns, symbol_exprt("b", bool_type()), float_type());
  }
  SECTION("float converts to Boolean (symbol)")
  {
    require_overloads_agree(ns, symbol_exprt("f", float_type()), bool_type());
  }
  SECTION("long double converts to long long (symbol)")
  {
    require_overloads_agree(
      ns, symbol_exprt("ld", long_double_type()), long_long_int_type());
  }
  SECTION("__int128 constant folds into int")
  {
    require_overloads_agree(ns, from_integer(1, int128_type()), int_type());
  }
  SECTION("narrowing fold truncates: 257 into signed char")
  {
    require_overloads_agree(
      ns, from_integer(257, int_type()), signed_char_type());
  }
  SECTION("unsigned char constant widens to long long")
  {
    require_overloads_agree(
      ns, from_integer(255, unsigned_char_type()), long_long_int_type());
  }
}

TEST_CASE(
  "both c_implicit_typecast overloads agree on further pointer shapes",
  "[c_typecast]")
{
  contextt ctx;
  namespacet ns(ctx);

  SECTION("void pointer converts to object pointer")
  {
    require_overloads_agree(
      ns,
      symbol_exprt("v", pointer_typet(empty_typet())),
      pointer_typet(int_type()));
  }
  SECTION("pointer-to-pointer converts to pointer (generous scalar rule)")
  {
    require_overloads_agree(
      ns,
      symbol_exprt("pp", pointer_typet(pointer_typet(int_type()))),
      pointer_typet(int_type()));
  }
  SECTION("pointer converts to Boolean")
  {
    require_overloads_agree(
      ns, symbol_exprt("p", pointer_typet(int_type())), bool_type());
  }
}

TEST_CASE(
  "both c_implicit_typecast overloads agree on pointer conversions",
  "[c_typecast]")
{
  contextt ctx;
  namespacet ns(ctx);

  // The two copies spell the null pointer differently -- the irept one builds a
  // constant whose value is "NULL", the expr2tc one a symbol named "NULL" --
  // and migrate_expr reconciles them (migrate.cpp:810). Left unpinned, a change
  // to either spelling would go unnoticed until a frontend compared pointers.
  SECTION("literal zero becomes the null pointer")
  {
    require_overloads_agree(
      ns, from_integer(0, int_type()), pointer_typet(int_type()));
  }

  SECTION("pointer converts to void pointer")
  {
    require_overloads_agree(
      ns,
      symbol_exprt("p", pointer_typet(int_type())),
      pointer_typet(empty_typet()));
  }

  SECTION("pointer converts to an unrelated pointer type")
  {
    require_overloads_agree(
      ns,
      symbol_exprt("p", pointer_typet(int_type())),
      pointer_typet(char_type()));
  }

  SECTION("pointer conversion to its own type is a no-op")
  {
    require_overloads_agree(
      ns,
      symbol_exprt("p", pointer_typet(int_type())),
      pointer_typet(int_type()));
  }
}

TEST_CASE(
  "both c_implicit_typecast_arithmetic overloads agree on operand pairs",
  "[c_typecast]")
{
  contextt ctx;
  namespacet ns(ctx);

  SECTION("sub-int types promote to int")
  {
    require_arith_result(
      ns,
      symbol_exprt("a", signed_char_type()),
      symbol_exprt("b", signed_char_type()),
      int_type());
  }
  SECTION("int and double promote to double")
  {
    require_arith_result(
      ns,
      symbol_exprt("i", int_type()),
      symbol_exprt("d", double_type()),
      double_type());
  }
  SECTION("signed and unsigned int promote to unsigned int")
  {
    require_arith_result(
      ns,
      symbol_exprt("s", int_type()),
      symbol_exprt("u", uint_type()),
      uint_type());
  }
  SECTION("Boolean and signed char promote to int")
  {
    require_arith_result(
      ns,
      symbol_exprt("b", bool_type()),
      symbol_exprt("c", signed_char_type()),
      int_type());
  }
  SECTION("float and long long promote to float")
  {
    require_arith_result(
      ns,
      symbol_exprt("f", float_type()),
      symbol_exprt("l", long_long_int_type()),
      float_type());
  }
  SECTION("constant char pair promotes and folds to int")
  {
    require_arith_result(
      ns,
      from_integer(1, signed_char_type()),
      from_integer(2, signed_char_type()),
      int_type());
  }
  SECTION("long long and __int128 promote to __int128")
  {
    require_arith_result(
      ns,
      symbol_exprt("l", long_long_int_type()),
      symbol_exprt("w", int128_type()),
      int128_type());
  }
  SECTION("int and unsigned __int128 promote to unsigned __int128")
  {
    require_arith_result(
      ns,
      symbol_exprt("i", int_type()),
      symbol_exprt("w", uint128_type()),
      uint128_type());
  }
  SECTION("void pointer and object pointer reconcile to the object pointer")
  {
    require_arith_result(
      ns,
      symbol_exprt("v", pointer_typet(empty_typet())),
      symbol_exprt("p", pointer_typet(int_type())),
      pointer_typet(int_type()));
  }
  SECTION("array decays to its element pointer against a pointer operand")
  {
    const array_typet arr(int_type(), from_integer(4, index_type()));
    require_arith_result(
      ns,
      symbol_exprt("a", arr),
      symbol_exprt("p", pointer_typet(int_type())),
      pointer_typet(int_type()));
  }
  SECTION("float and __int128 promote to float")
  {
    // C17 6.3.1.8: with one floating operand, both convert to the floating
    // type -- integer rank never outranks it, however wide the integer.
    require_arith_result(
      ns,
      symbol_exprt("f", float_type()),
      symbol_exprt("w", int128_type()),
      float_type());
  }
  SECTION("double and __int128 promote to double")
  {
    require_arith_result(
      ns,
      symbol_exprt("d", double_type()),
      symbol_exprt("w", int128_type()),
      double_type());
  }
  SECTION("pointer and __int128 convert neither operand")
  {
    // C17 6.5.6: pointer arithmetic converts neither operand -- 6.3.1.8
    // covers arithmetic types only. (pointer, int) already gets this
    // treatment; a wider integer must not change it.
    const exprt ptr = symbol_exprt("p", pointer_typet(int_type()));
    const exprt wide = symbol_exprt("w", int128_type());
    require_arith_operands(ns, ptr, wide, ptr.type(), wide.type());
  }
}

TEST_CASE(
  "both c_implicit_typecast overloads agree over fixed-point types",
  "[c_typecast]")
{
  contextt ctx;
  namespacet ns(ctx);

  // c_typecast.h ranks FIXED between the integers and the floats, so a
  // fixed-point operand outranks any integer and yields to any float. The
  // integer side does not adopt the fixedbv operand's format: it becomes a
  // zero-fraction fixedbv of its own width, so the arithmetic happens without
  // discarding any of its value.
  SECTION("int becomes a zero-fraction fixedbv of its own width")
  {
    require_legacy_arith_operands(
      ns,
      symbol_exprt("f", accum_type()),
      symbol_exprt("i", int_type()),
      accum_type(),
      fixedbv_type(32, 32, true));
  }
  SECTION("two fixedbv operands are left in their own formats")
  {
    // fixed op fixed returns early: the operation is computed in the
    // operands' common format, so neither side is cast (c_typecast.cpp).
    require_legacy_arith_operands(
      ns,
      symbol_exprt("r", fract_type()),
      symbol_exprt("a", accum_type()),
      fract_type(),
      accum_type());
  }
  SECTION("a 128-bit integer keeps its width, and the fixedbv is untouched")
  {
    // Integer rank, however wide, never outranks a fixed-point type, so the
    // fixedbv operand is left alone rather than widened to 128 bits.
    require_legacy_arith_operands(
      ns,
      symbol_exprt("f", accum_type()),
      symbol_exprt("w", int128_type()),
      accum_type(),
      fixedbv_type(128, 128, true));
  }
  SECTION("fixedbv yields to a floating type")
  {
    require_arith_result(
      ns,
      symbol_exprt("f", accum_type()),
      symbol_exprt("d", double_type()),
      double_type());
  }
  SECTION("int constant folds into a fixedbv destination")
  {
    require_overloads_agree(ns, from_integer(1, int_type()), accum_type());
  }
  SECTION("fixedbv symbol converts to int")
  {
    require_overloads_agree(ns, symbol_exprt("f", accum_type()), int_type());
  }
}
