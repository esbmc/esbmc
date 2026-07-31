#include <pointer-analysis/andersen.h>

#include <irep2/irep2_utils.h>
#include <util/c_types.h>

/// \file
/// Andersen points-to analysis — skeleton.
///
/// The plumbing (node interning, constraint/edge storage, and the query
/// translation) is implemented for you.  The two methods that *are* the
/// algorithm — \ref andersent::solve and \ref andersent::collect_constraints —
/// are left as guided TODOs.  The unit tests in
/// `unit/pointer-analysis/andersen.test.cpp` specify exactly what a correct
/// \ref andersent::solve must produce.

andersent::andersent()
{
  // Reserve node 0 as TOP ("may point anywhere").  It owns a slot in every
  // node-indexed vector but is never handed out by get_node, so real program
  // nodes are numbered from 1.  See andersen.h::andersen_soundness.
  node_to_expr.push_back(unknown2tc(pointer_type2()));
  ensure_node(TOP);
}

void andersent::ensure_node(node_id n)
{
  if (n >= pts.size())
  {
    pts.resize(n + 1);
    copy_edges.resize(n + 1);
  }
}

void andersent::points_to_top(node_id n)
{
  ensure_node(n);
  pts[n].insert(TOP);
}

andersent::node_id andersent::get_node(const expr2tc &e)
{
  auto it = expr_to_node.find(e);
  if (it != expr_to_node.end())
    return it->second;

  node_id n = node_to_expr.size();
  expr_to_node.emplace(e, n);
  node_to_expr.push_back(e);
  ensure_node(n);
  return n;
}

void andersent::add_constraint(constraint_kindt kind, node_id lhs, node_id rhs)
{
  ensure_node(lhs);
  ensure_node(rhs);
  constraints.push_back(constraintt{kind, lhs, rhs});
  solved = false;
}

void andersent::solve()
{
  // ==========================================================================
  // EXERCISE 1 — the core Andersen fixpoint.
  //
  // Goal: populate `pts` so that, for every registered constraint, the
  // set-inclusion in andersen.h::andersen_constraints holds.
  //
  // Suggested implementation (dynamic transitive closure, see
  // andersen.h::andersen_solve):
  //
  //   1. SEED.  For each ADDRESS_OF constraint {lhs, rhs}:
  //          pts[lhs].insert(rhs);
  //      For each COPY constraint {lhs, rhs}:
  //          copy_edges[rhs].push_back(lhs);   // edge rhs -> lhs
  //      Put every node whose pts is non-empty on the work-list.
  //
  //   2. PROPAGATE.  While the work-list is non-empty, pop a node `n`:
  //
  //        a) Resolve dynamic constraints that read/write through `n`:
  //             for each LOAD  {p, q} with q == n:                 // p = *n
  //                 for each o in pts[n]: add copy edge  o -> p
  //             for each STORE {p, q} with p == n:                 // *n = q
  //                 for each o in pts[n]: add copy edge  q -> o
  //           (When you add a copy edge x -> y, immediately union pts[x] into
  //            pts[y] and, if pts[y] grew, queue y.)
  //
  //        b) Push along existing copy edges out of `n`:
  //             for each m in copy_edges[n]:
  //                 if (union pts[n] into pts[m] changed pts[m]) queue m;
  //
  //      Iterating linearly over `constraints` in (a) is O(constraints) per
  //      pop — correct but slow.  Once the tests pass, index LOAD/STORE
  //      constraints by their pointer node to speed this up.
  //
  //   3. Terminates because sets only grow and are bounded by the node count.
  //
  // TOP must stay absorbing (see andersen.h::andersen_soundness).  When `n`
  // may point anywhere (TOP in pts[n]), a dereference through `n` cannot be
  // resolved to specific objects, so in step (a):
  //   - LOAD  p = *n  with TOP in pts[n]:  p may now point anywhere too ->
  //                                        pts[p].insert(TOP)  (queue p)
  //   - STORE *n = q  with TOP in pts[n]:  q flows to an unknown object; the
  //                                        conservative sink is pts[TOP] |= pts[q].
  // Do NOT try to add copy edges to/from the objects "inside" TOP — there are
  // none; the whole point is that they are unnameable.  Propagating TOP like
  // any other element along copy edges (step b) already keeps it flowing
  // forward; these two rules just stop a TOP pointer from being silently
  // treated as pointing to nothing under * .
  //
  // Tip: a small helper `bool join(dst, src)` that unions `src` into `pts[dst]`
  // and returns whether it changed keeps the loop readable.
  // ==========================================================================

  solved = true;
}

const std::unordered_set<andersent::node_id> &
andersent::points_to(node_id n) const
{
  static const std::unordered_set<node_id> empty;
  if (n >= pts.size())
    return empty;
  return pts[n];
}

bool andersent::may_point_to(node_id a, node_id b) const
{
  const auto &s = points_to(a);
  return s.find(b) != s.end();
}

void andersent::collect_constraints(const goto_functionst &goto_functions)
{
  // ==========================================================================
  // EXERCISE 2 — lower the program to constraints.
  //
  // Walk every reachable ASSIGN (and treat FUNCTION_CALL argument/return
  // binding as assignments) and classify the shape of `lhs := rhs`:
  //
  //   lhs = &a         -> add_constraint(ADDRESS_OF, node(lhs), node(a))
  //   lhs = q          -> add_constraint(COPY,       node(lhs), node(q))
  //   lhs = *q         -> add_constraint(LOAD,       node(lhs), node(q))
  //   *p  = rhs        -> add_constraint(STORE,      node(p),   node(rhs))
  //
  // Practicalities to decide as you go:
  //   - Field sensitivity: model `s.f` as its own node (name+suffix) or fold
  //     all fields of `s` into one node (field-insensitive) to start simple.
  //   - Arrays: fold every element into the array's node (index-insensitive).
  //   - Heap: intern one node per allocation site (the malloc/new location).
  //   - Only pointer-typed assignments produce constraints; skip the rest.
  //
  // SOUNDNESS: if a pointer LHS gets its value in a way you do NOT model
  // (external/undefined call, inline asm, int->ptr cast, union pun, varargs),
  // you must NOT leave its set empty — call points_to_top(node(lhs)) so the
  // pointer conservatively "may point anywhere".  An empty set is an unsound
  // under-approximation; TOP keeps model-less code safe.  See
  // andersen.h::andersen_soundness.
  //
  // Keep this frontend independent of solve(): it just builds constraints.
  // ==========================================================================
  (void)goto_functions;
}

void andersent::operator()(const goto_functionst &goto_functions)
{
  collect_constraints(goto_functions);
  solve();
}

void andersent::to_object_descriptors(node_id n, valuest &dest) const
{
  const auto &set = points_to(n);

  // TOP subsumes everything: report a single unnameable target so consumers
  // (is_object_descriptor2t checks) abstain / havoc conservatively.
  if (set.count(TOP))
  {
    dest.push_back(unknown2tc(pointer_type2()));
    return;
  }

  for (node_id pointee : set)
  {
    const expr2tc &obj = node_to_expr[pointee];
    dest.push_back(
      object_descriptor2tc(obj->type, obj, gen_zero(index_type2()), 0));
  }
}

void andersent::get_values(
  goto_programt::const_targett,
  const expr2tc &expr,
  valuest &dest)
{
  auto it = expr_to_node.find(expr);
  if (it != expr_to_node.end())
    to_object_descriptors(it->second, dest);
}

void andersent::get_reference_set(const expr2tc &expr, valuest &dest)
{
  // For a dereference *p, the referenced objects are exactly pts[p].
  expr2tc pointer = expr;
  if (is_dereference2t(expr))
    pointer = to_dereference2t(expr).value;

  auto it = expr_to_node.find(pointer);
  if (it != expr_to_node.end())
    to_object_descriptors(it->second, dest);
}
