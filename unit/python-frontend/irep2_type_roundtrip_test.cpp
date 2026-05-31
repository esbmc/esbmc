#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/global_scope.h>
#include <util/config.h>
#include <util/context.h>
#include <util/migrate.h>
#include <util/c_types.h>
#include <util/python_types.h>
#include <util/arith_tools.h>

// Phase 4.0 equivalence harness for the Python frontend → IREP2 migration
// (docs/irep2-migration.md Part IV §7 Phase 4.0, item 2).
//
// This is the *durable contract regression* that Phases 4.3/4.4 validate
// against: it pins that every typet the Python frontend builds today survives
// the legacy↔IREP2 boundary losslessly. The contract is the same lossless-cache
// property unit/util/migrate.test.cpp pins for synthetic IREP2 nodes (B2,
// esbmc/esbmc#4715), specialised here to the *exact corpus* type_handler emits.
//
// Two levels are checked:
//
//  (1) IREP2-side stability (the hard contract):
//        migrate_type(migrate_type_back(migrate_type(t))) == migrate_type(t)
//      We assert IREP2 round-trip — not legacy byte-equality — because
//      migrate_type deliberately canonicalises the legacy form (migrate.test.cpp
//      §header). This is precisely what makes the symbol-store IREP2 cache
//      lossless, and what Phase 4.3 must not break when it builds type2tc
//      directly and back-migrates only at the create_symbol seam.
//
//  (2) Migration-critical legacy invariants the plan calls out explicitly:
//        - python_int_typet() width is identical in *both* int-encoding modes
//          (RP4 / F-P7 / SM1): the 64-bit default and the 512-bit --ir bignum.
//        - the complex struct_type round-trips component-for-component (Q-P5).
//        - the str/char "#cpp_type" seam attribute (F-P5): its handling across
//          the boundary is pinned so Phase 4.5's re-attachment seam is checked.

namespace
{
// (1) IREP2 representation of a Python type is stable under a legacy round-trip.
void require_irep2_stable(const typet &t)
{
  const type2tc t2 = migrate_type(t);
  const type2tc back = migrate_type(migrate_type_back(t2));
  REQUIRE(back == t2);
}
} // namespace

TEST_CASE(
  "python type corpus is IREP2-round-trip stable",
  "[python-frontend][irep2][migrate]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  // The scalar/composite typet corpus type_handler::get_typet emits, built the
  // same way the frontend builds it (see type_handler.cpp get_typet(string)).
  SECTION("scalar kinds")
  {
    require_irep2_stable(bool_type());           // bool
    require_irep2_stable(long_long_int_type());  // int / ord / abs
    require_irep2_stable(long_long_uint_type()); // uint / uint64 / Epoch / Slot
    require_irep2_stable(double_type());         // float
    require_irep2_stable(uint256_type());        // uint256 / BLSFieldElement
    require_irep2_stable(char_type());           // str/chr, size == 1
    require_irep2_stable(empty_typet()); // tuple / divmod / "" annotation
    require_irep2_stable(any_type());    // Any / object
  }

  SECTION("pointer kinds")
  {
    require_irep2_stable(pointer_type()); // NoneType / Optional

    code_typet code_type; // Callable
    code_type.return_type() = empty_typet();
    require_irep2_stable(pointer_typet(code_type));
  }

  SECTION("array kinds (str / bytes / type)")
  {
    // build_array(char_type(), N): array of chars backing str/bytes/type.
    require_irep2_stable(
      array_typet(char_type(), from_integer(10, size_type())));
    require_irep2_stable(
      array_typet(long_long_int_type(), from_integer(4, size_type()))); // bytes
  }

  SECTION("complex struct (Q-P5)")
  {
    require_irep2_stable(get_complex_struct_type());
  }
}

TEST_CASE(
  "python_int_typet width is preserved across the boundary (RP4)",
  "[python-frontend][irep2][migrate]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  // Default mode: 64-bit signed (long_long_int_type()).
  config.options.set_option("int-encoding", false);
  {
    const typet wide = type_handler::python_int_typet();
    const typet back = migrate_type_back(migrate_type(wide));
    REQUIRE(back.is_signedbv());
    REQUIRE(
      to_signedbv_type(back).get_width() == to_signedbv_type(wide).get_width());
  }

  // --ir / int-encoding mode: 512-bit bignum (kPythonBignumWidth). This width
  // is what F-P7/SM1 forbid the migration from silently changing.
  config.options.set_option("int-encoding", true);
  {
    const typet wide = type_handler::python_int_typet();
    REQUIRE(wide.is_signedbv());
    const typet back = migrate_type_back(migrate_type(wide));
    REQUIRE(back.is_signedbv());
    REQUIRE(
      to_signedbv_type(back).get_width() == to_signedbv_type(wide).get_width());
  }

  config.options.set_option("int-encoding", false);
}

TEST_CASE(
  "complex struct components survive the boundary (Q-P5)",
  "[python-frontend][irep2][migrate]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  const struct_typet complex_type = get_complex_struct_type();
  const typet back = migrate_type_back(migrate_type(complex_type));

  REQUIRE(back.is_struct());
  const struct_typet &back_struct = to_struct_type(back);
  REQUIRE(back_struct.components().size() == 2);
  REQUIRE(back_struct.components()[0].get_name() == "real");
  REQUIRE(back_struct.components()[1].get_name() == "imag");
  REQUIRE(back_struct.components()[0].type() == double_type());
  REQUIRE(back_struct.components()[1].type() == double_type());
  REQUIRE(is_complex_type(back));
}

TEST_CASE(
  "str/char #cpp_type seam attribute handling (F-P5)",
  "[python-frontend][irep2][migrate]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  // type_handler::get_typet(str, /*size*/1) tags an 8-bit char with #cpp_type
  // "char" for C-backend compatibility. This attribute is read by shared
  // downstream passes (F-P5), so the plan keeps it on the *legacy seam node*
  // rather than migrating it. This test pins how it behaves across the boundary.
  typet tagged = char_type();
  type_utils::set_cpp_type(tagged, "char");
  REQUIRE(type_utils::get_cpp_type(tagged) == "char");

  const typet back = migrate_type_back(migrate_type(tagged));

  // The underlying bit-vector (signedness + width) always survives.
  REQUIRE(back.get_bool("#signed") == tagged.get_bool("#signed"));
  REQUIRE(back.width() == tagged.width());

  // Pinned finding (the reason F-P5 is a seam, not a migration target):
  // migrate_type/migrate_type_back DROP #cpp_type — IREP2 has no slot for it,
  // so a round-tripped char carries no #cpp_type. Therefore Phase 4.5 must
  // RE-ATTACH #cpp_type at the legacy hand-off; it cannot ride through IREP2.
  // If a future migrate.cpp change ever starts preserving it, this assertion
  // fires and the §15/Phase-4.5 re-attachment rationale must be revisited.
  REQUIRE(type_utils::get_cpp_type(back).empty());
  REQUIRE(
    type_utils::get_cpp_type(tagged) == "char"); // unchanged on the seam node
}

TEST_CASE(
  "Phase 4.3: elementary get_typet outputs are byte-identical (IREP2-internal)",
  "[python-frontend][irep2][phase4.3]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  contextt context;
  global_scope gs;
  const nlohmann::json ast = {
    {"_type", "Module"},
    {"body", nlohmann::json::array()},
    {"filename", "test.py"},
    {"type_ignores", nlohmann::json::array()}};
  python_converter converter(context, &ast, gs);
  const type_handler &th = converter.get_type_handler();

  // The elementary family now builds type2tc internally and lowers at the
  // seam; the legacy typet reaching create_symbol must be byte-identical to the
  // direct legacy builder it replaced.
  REQUIRE(th.get_typet(std::string("int")) == long_long_int_type());
  REQUIRE(
    th.get_typet(std::string("GeneralizedIndex")) == long_long_int_type());
  REQUIRE(th.get_typet(std::string("uint")) == long_long_uint_type());
  REQUIRE(th.get_typet(std::string("uint64")) == long_long_uint_type());
  REQUIRE(th.get_typet(std::string("Epoch")) == long_long_uint_type());
  REQUIRE(th.get_typet(std::string("bool")) == bool_type());
  REQUIRE(th.get_typet(std::string("uint256")) == uint256_type());
  REQUIRE(th.get_typet(std::string("BLSFieldElement")) == uint256_type());

  // str/size==1: an 8-bit char carrying #cpp_type "char", re-attached at the
  // seam (F-P5) since IREP2 cannot carry it. Both the bit-vector and the
  // re-attached hint must match the legacy builder byte-for-byte.
  typet expected_char = char_type();
  type_utils::set_cpp_type(expected_char, "char");
  const typet got_char = th.get_typet(std::string("str"), 1);
  REQUIRE(got_char == expected_char);
  REQUIRE(type_utils::get_cpp_type(got_char) == "char");
}
