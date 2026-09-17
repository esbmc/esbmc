/// \file
/// Specification tests for the core Andersen solver.
///
/// These tests drive andersent::solve() directly with hand-built constraints,
/// so they exercise the *algorithm* without needing a GOTO program.  Against
/// the skeleton (empty solve()) every "solve" test below FAILS — that is
/// intentional.  Implement andersent::solve() until they all pass; the
/// assertions are the specification of a correct fixpoint.
///
/// Run just these:
///   ctest -R andersen-test --output-on-failure

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <pointer-analysis/andersen.h>
#include <util/lang/c_types.h>
#include <irep2/irep2_utils.h>

namespace
{
/// A distinct nameable object per test variable, so get_node() interns a
/// stable node.  The concrete expr is irrelevant to the core algorithm; only
/// its identity matters.
expr2tc make_var(const std::string &name)
{
  return symbol2tc(pointer_type2(), irep_idt(name));
}
} // namespace

TEST_CASE("node interning is stable", "[andersen]")
{
  andersent a;
  expr2tc p = make_var("p");
  expr2tc q = make_var("q");

  auto np = a.get_node(p);
  auto nq = a.get_node(q);

  REQUIRE(np != nq);
  REQUIRE(a.get_node(p) == np);             // same expr -> same node
  REQUIRE(a.get_node(make_var("p")) == np); // equal expr -> same node
}

TEST_CASE("symbols are interned by identifier", "[andersen]")
{
  // A formal is interned from the callee's signature at the call site and from
  // the body's own symbol inside it.  Those two spellings of the type need not
  // match (a symbol_type against the struct it names, say), and if they picked
  // different nodes the argument binding would target neither.
  andersent a;

  auto by_pointer = a.get_node(symbol2tc(pointer_type2(), "p"));
  auto by_int = a.get_node(symbol2tc(int_type2(), "p"));

  REQUIRE(by_pointer == by_int);
}

TEST_CASE("address-of then copy", "[andersen][solve]")
{
  // p = &a;  q = &b;  p = q;   =>  pts(p) = {a,b}, pts(q) = {b}
  andersent a;
  auto np = a.get_node(make_var("p"));
  auto nq = a.get_node(make_var("q"));
  auto na = a.get_node(make_var("a"));
  auto nb = a.get_node(make_var("b"));

  a.add_constraint(andersent::constraint_kindt::ADDRESS_OF, np, na);
  a.add_constraint(andersent::constraint_kindt::ADDRESS_OF, nq, nb);
  a.add_constraint(andersent::constraint_kindt::COPY, np, nq);

  a.solve();

  REQUIRE(a.points_to(np) == std::unordered_set<andersent::node_id>{na, nb});
  REQUIRE(a.points_to(nq) == std::unordered_set<andersent::node_id>{nb});
}

TEST_CASE("copy is transitive", "[andersen][solve]")
{
  // a = &o;  b = a;  c = b;   =>  pts(a)=pts(b)=pts(c)={o}
  andersent a;
  auto na = a.get_node(make_var("a"));
  auto nb = a.get_node(make_var("b"));
  auto nc = a.get_node(make_var("c"));
  auto no = a.get_node(make_var("o"));

  a.add_constraint(andersent::constraint_kindt::ADDRESS_OF, na, no);
  a.add_constraint(andersent::constraint_kindt::COPY, nb, na);
  a.add_constraint(andersent::constraint_kindt::COPY, nc, nb);

  a.solve();

  REQUIRE(a.may_point_to(nb, no));
  REQUIRE(a.may_point_to(nc, no));
}

TEST_CASE("load resolves through the pointer", "[andersen][solve]")
{
  // x = &y;  y = &z;  w = *x;   =>  pts(w) = {z}
  andersent a;
  auto nx = a.get_node(make_var("x"));
  auto ny = a.get_node(make_var("y"));
  auto nz = a.get_node(make_var("z"));
  auto nw = a.get_node(make_var("w"));

  a.add_constraint(andersent::constraint_kindt::ADDRESS_OF, nx, ny);
  a.add_constraint(andersent::constraint_kindt::ADDRESS_OF, ny, nz);
  a.add_constraint(andersent::constraint_kindt::LOAD, nw, nx); // w = *x

  a.solve();

  REQUIRE(a.points_to(nw) == std::unordered_set<andersent::node_id>{nz});
}

TEST_CASE("store writes through the pointer", "[andersen][solve]")
{
  // x = &y;  z = &w;  *x = z;   =>  pts(y) = {w}
  andersent a;
  auto nx = a.get_node(make_var("x"));
  auto ny = a.get_node(make_var("y"));
  auto nz = a.get_node(make_var("z"));
  auto nw = a.get_node(make_var("w"));

  a.add_constraint(andersent::constraint_kindt::ADDRESS_OF, nx, ny);
  a.add_constraint(andersent::constraint_kindt::ADDRESS_OF, nz, nw);
  a.add_constraint(andersent::constraint_kindt::STORE, nx, nz); // *x = z

  a.solve();

  REQUIRE(a.points_to(ny) == std::unordered_set<andersent::node_id>{nw});
}

TEST_CASE("classic mixed example", "[andersen][solve]")
{
  // p=&a; q=&b; p=q; r=&p; s=*r; *r=q;
  //   => pts(p)={a,b}, pts(q)={b}, pts(r)={p}, pts(s)={a,b}
  andersent a;
  auto np = a.get_node(make_var("p"));
  auto nq = a.get_node(make_var("q"));
  auto nr = a.get_node(make_var("r"));
  auto ns = a.get_node(make_var("s"));
  auto na = a.get_node(make_var("a"));
  auto nb = a.get_node(make_var("b"));

  using k = andersent::constraint_kindt;
  a.add_constraint(k::ADDRESS_OF, np, na);
  a.add_constraint(k::ADDRESS_OF, nq, nb);
  a.add_constraint(k::COPY, np, nq);
  a.add_constraint(k::ADDRESS_OF, nr, np);
  a.add_constraint(k::LOAD, ns, nr);  // s = *r
  a.add_constraint(k::STORE, nr, nq); // *r = q

  a.solve();

  using set = std::unordered_set<andersent::node_id>;
  REQUIRE(a.points_to(np) == set{na, nb});
  REQUIRE(a.points_to(nq) == set{nb});
  REQUIRE(a.points_to(nr) == set{np});
  REQUIRE(a.points_to(ns) == set{na, nb});
}

TEST_CASE("TOP reports an unnameable target so consumers abstain", "[andersen]")
{
  // A pointer the frontend cannot model must point to TOP, and the query layer
  // must surface a non-object-descriptor (unknown) rather than an empty set.
  // Exercises the soundness escape hatch directly; no solve() needed.
  andersent a;
  expr2tc p = make_var("p");
  a.points_to_top(a.get_node(p));

  value_setst::valuest values;
  a.get_values(goto_programt::const_targett(), p, values);

  REQUIRE(values.size() == 1);
  REQUIRE_FALSE(is_object_descriptor2t(values.front())); // consumers abstain
}

TEST_CASE("query layer yields object descriptors", "[andersen][solve]")
{
  // p = &a;  =>  get_values(p) contains object_descriptor(a)
  andersent a;
  expr2tc p = make_var("p");
  expr2tc obj_a = make_var("a");
  a.add_constraint(
    andersent::constraint_kindt::ADDRESS_OF, a.get_node(p), a.get_node(obj_a));

  a.solve();

  value_setst::valuest values;
  a.get_values(goto_programt::const_targett(), p, values);

  REQUIRE(values.size() == 1);
  REQUIRE(is_object_descriptor2t(values.front()));
  REQUIRE(to_object_descriptor2t(values.front()).object == obj_a);
}
