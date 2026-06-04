/*******************************************************************
 Module: exception_typeidt unit tests (issue #5075, phase P0)

 Test plan: build small C++ class hierarchies, then check that the
 closed-world type-id registry assigns stable unique ids and that its
 reflexive-transitive subtype closure matches the hierarchy — including
 transitive, multiple, and diamond inheritance, which the symbolic
 catch-match guard relies on.

\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../testing-utils/goto_factory.h"
#include <goto-programs/exception_typeid.h>
#include <goto-programs/exception_globals.h>
#include <util/context.h>
#include <util/symbol.h>

namespace
{
/// Parse @p code as C++ and build the type-id registry over its symbols.
program build(std::string code)
{
  return goto_factory::get_goto_functions(
    code, goto_factory::Architecture::BIT_64, "test.cpp");
}
} // namespace

TEST_CASE("single-chain subtype closure", "[exception_typeid]")
{
  std::string code = "struct A { int a; };\n"
                     "struct B : A { int b; };\n"
                     "struct C : B { int c; };\n"
                     "struct X { int x; };\n"
                     "int main() { A a; B b; C c; X x; return 0; }\n";
  program P = build(code);
  exception_typeidt reg(P.ns);

  // Reflexive.
  REQUIRE(reg.is_subtype("A", "A"));
  REQUIRE(reg.is_subtype("X", "X"));

  // Direct and transitive derivation.
  REQUIRE(reg.is_subtype("B", "A"));
  REQUIRE(reg.is_subtype("C", "B"));
  REQUIRE(reg.is_subtype("C", "A")); // transitive

  // Not symmetric: a base is not a subtype of its derived class.
  REQUIRE_FALSE(reg.is_subtype("A", "B"));
  REQUIRE_FALSE(reg.is_subtype("A", "C"));

  // Unrelated hierarchy.
  REQUIRE_FALSE(reg.is_subtype("X", "A"));
  REQUIRE_FALSE(reg.is_subtype("A", "X"));

  // The catch-match set for `catch (A)` is exactly {A,B,C} (not X).
  std::set<unsigned> caught_by_A = reg.concrete_subtype_ids("A");
  REQUIRE(caught_by_A.count(reg.id_of("A")) == 1);
  REQUIRE(caught_by_A.count(reg.id_of("B")) == 1);
  REQUIRE(caught_by_A.count(reg.id_of("C")) == 1);
  REQUIRE(caught_by_A.count(reg.id_of("X")) == 0);

  // `catch (X)` matches only X.
  std::set<unsigned> caught_by_X = reg.concrete_subtype_ids("X");
  REQUIRE(caught_by_X == std::set<unsigned>{reg.id_of("X")});
}

TEST_CASE("ids are unique, stable, and non-zero", "[exception_typeid]")
{
  std::string code = "struct A { int a; };\n"
                     "struct B : A { int b; };\n"
                     "int main() { A a; B b; return 0; }\n";
  program P = build(code);
  exception_typeidt reg(P.ns);

  unsigned ia = reg.id_of("A");
  unsigned ib = reg.id_of("B");
  REQUIRE(ia != exception_typeidt::no_type);
  REQUIRE(ib != exception_typeidt::no_type);
  REQUIRE(ia != ib);

  // Stable across repeated queries.
  REQUIRE(reg.id_of("A") == ia);
  REQUIRE(reg.id_of("B") == ib);

  // An unknown (opaque) name gets a fresh, distinct id rather than colliding.
  unsigned iu = reg.id_of("NotAProgramType");
  REQUIRE(iu != exception_typeidt::no_type);
  REQUIRE(iu != ia);
  REQUIRE(iu != ib);
  REQUIRE(reg.id_of("NotAProgramType") == iu); // stable once assigned
}

TEST_CASE("exception-state globals are created idempotently", "[exception_typeid]")
{
  std::string code = "int main() { return 0; }\n";
  program P = build(code);

  REQUIRE(P.context.find_symbol(exception_globals::thrown_id) == nullptr);

  create_exception_state_symbols(P.context);

  const symbolt *thrown = P.context.find_symbol(exception_globals::thrown_id);
  const symbolt *tid = P.context.find_symbol(exception_globals::typeid_id);
  const symbolt *val = P.context.find_symbol(exception_globals::value_id);
  REQUIRE(thrown != nullptr);
  REQUIRE(tid != nullptr);
  REQUIRE(val != nullptr);

  // Right shapes: bool thrown flag, pointer value slot, static lifetime.
  REQUIRE(thrown->get_type().id() == "bool");
  REQUIRE(val->get_type().id() == "pointer");
  REQUIRE(thrown->static_lifetime);
  REQUIRE(tid->static_lifetime);
  REQUIRE(val->static_lifetime);

  // Second call is a no-op (no duplicate / no crash).
  create_exception_state_symbols(P.context);
  REQUIRE(P.context.find_symbol(exception_globals::thrown_id) == thrown);
}

TEST_CASE("multiple and diamond inheritance", "[exception_typeid]")
{
  // D derives from A through two paths (B and C); A is a shared base.
  std::string code = "struct A { int a; };\n"
                     "struct B : A { int b; };\n"
                     "struct C : A { int c; };\n"
                     "struct D : B, C { int d; };\n"
                     "int main() { A a; B b; C c; D d; return 0; }\n";
  program P = build(code);
  exception_typeidt reg(P.ns);

  // D is a subtype of every ancestor, reached through both diamond arms,
  // and the cycle-guarded walk terminates.
  REQUIRE(reg.is_subtype("D", "B"));
  REQUIRE(reg.is_subtype("D", "C"));
  REQUIRE(reg.is_subtype("D", "A"));
  REQUIRE(reg.is_subtype("B", "A"));
  REQUIRE(reg.is_subtype("C", "A"));

  // A `catch (A)` catches the whole hierarchy.
  std::set<unsigned> caught_by_A = reg.concrete_subtype_ids("A");
  REQUIRE(caught_by_A.count(reg.id_of("A")) == 1);
  REQUIRE(caught_by_A.count(reg.id_of("B")) == 1);
  REQUIRE(caught_by_A.count(reg.id_of("C")) == 1);
  REQUIRE(caught_by_A.count(reg.id_of("D")) == 1);

  // `catch (B)` catches B and D, but not its sibling C or shared base A.
  std::set<unsigned> caught_by_B = reg.concrete_subtype_ids("B");
  REQUIRE(caught_by_B.count(reg.id_of("B")) == 1);
  REQUIRE(caught_by_B.count(reg.id_of("D")) == 1);
  REQUIRE(caught_by_B.count(reg.id_of("C")) == 0);
  REQUIRE(caught_by_B.count(reg.id_of("A")) == 0);
}
