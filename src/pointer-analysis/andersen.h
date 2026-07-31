#ifndef CPROVER_POINTER_ANALYSIS_ANDERSEN_H
#define CPROVER_POINTER_ANALYSIS_ANDERSEN_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <goto-programs/goto_functions.h>
#include <irep2/irep2.h>
#include <pointer-analysis/value_sets.h>

/// \file
/// Andersen-style (inclusion-based) points-to analysis.
///
/// \section andersen_what What this analysis computes
///
/// A *points-to* analysis answers the question "which objects can this
/// pointer refer to?".  Andersen's analysis computes a single,
/// **flow-insensitive**, **context-insensitive**, **whole-program**
/// answer: for every abstract location it gives one points-to set that is
/// valid at *every* program point.  It ignores statement order (flow
/// insensitivity) and merges all call sites of a function (context
/// insensitivity).  That makes it far cheaper than the flow-sensitive
/// value-set analysis, at the price of precision.
///
/// It is *sound*: the real target of any pointer at run time is always
/// contained in the computed set (assuming the constraint frontend models
/// every assignment).  It may be *imprecise*: the set can contain objects a
/// pointer never actually refers to.
///
/// \section andersen_model The abstraction: nodes
///
/// Every variable and every allocation site in the program is abstracted to
/// a single **node** (see \ref node_id).  A heap allocation inside a loop is
/// one node no matter how many concrete objects it creates at run time —
/// that is the source of most of Andersen's imprecision, and it is what keeps
/// the analysis finite and fast.
///
/// The result is a map `pts : node_id -> set<node_id>` where `n' in pts[n]`
/// means "the pointer abstracted by `n` may hold the address of the object
/// abstracted by `n'`".
///
/// \section andersen_constraints The four constraint kinds
///
/// Andersen reduces the whole program to a set of **set-inclusion (subset)
/// constraints**.  Every pointer-relevant assignment maps to exactly one of
/// four shapes.  Using `p`, `q` for pointer variables and `&a` for the
/// address of object `a`:
///
/// | # | Program statement | Constraint     | Meaning                          |
/// |---|-------------------|----------------|----------------------------------|
/// | 1 | `p = &a;`         | `pts[p] ⊇ {a}` | base / address-of                |
/// | 2 | `p = q;`          | `pts[p] ⊇ pts[q]` | copy                          |
/// | 3 | `p = *q;`         | `∀o∈pts[q]. pts[p] ⊇ pts[o]` | load               |
/// | 4 | `*p = q;`         | `∀o∈pts[p]. pts[o] ⊇ pts[q]` | store              |
///
/// Constraints 1 and 2 are *static*: they never change.  Constraints 3 and 4
/// are *dynamic*: the pairs they relate depend on the points-to sets, which
/// are still being computed — that is why solving needs a fixpoint.
///
/// \section andersen_solve Solving = dynamic transitive closure
///
/// Think of copy constraints as edges of a graph over nodes: a copy
/// `pts[p] ⊇ pts[q]` is an edge `q -> p` meaning "everything q points to also
/// flows to p".  Solving is:
///
///   1. Seed: apply every address-of constraint (`pts[p] ∪= {a}`).
///   2. Add a copy edge `q -> p` for every copy constraint.
///   3. Propagate to a fixpoint with a work-list of nodes whose set changed.
///      When node `n` is processed:
///        - Load  `p = *n`:   for each `o ∈ pts[n]`, add copy edge `o -> p`.
///        - Store `*n = q`:   for each `o ∈ pts[n]`, add copy edge `q -> o`.
///        - Push along every copy edge `n -> m`: `pts[m] ∪= pts[n]`;
///          if `pts[m]` grew, (re)queue `m`.
///      Adding an edge can itself enlarge a set, so newly created edges feed
///      back into the work-list.  The process terminates because sets only
///      ever grow and are bounded by the (finite) node set.
///
/// This is the classic O(n^3) dynamic-transitive-closure formulation.  Real
/// implementations bolt on cycle elimination (HCD/LCD) and better data
/// structures; those are deliberately left out here so the core algorithm is
/// visible.
///
/// \section andersen_layers How this file is organised
///
/// The class is split into three layers so the *algorithm* can be learned and
/// unit-tested without dragging in the GOTO/irep2 machinery:
///
///   - **Core solver** — pure, operates only on \ref node_id and constraints.
///     This is the heart of Andersen and the part you implement (see
///     \ref andersent::solve).  It is fully exercised by the unit tests using
///     hand-built constraints; no GOTO program is needed.
///   - **GOTO frontend** — \ref andersent::collect_constraints walks the
///     goto-functions and lowers each assignment to one of the four
///     constraints.  This is ESBMC-specific glue, provided as a stub.
///   - **Query / transparency layer** — implements \ref value_setst so the
///     existing whole-program consumers (currently GCSE, and the k-induction
///     pointer-array-write resolver) can query it exactly as they queried the
///     value-set analysis.
///
/// \section andersen_soundness Soundness contract (why the TOP node exists)
///
/// This analysis is deliberately the *coarsest, quickest* one: it exists only
/// to authorise **safe** GOTO rewrites, while symbolic execution provides the
/// real precision.  For that role correctness needs exactly one property: the
/// points-to set must be a sound **over-approximation** — a superset of what a
/// pointer can really hold.  Every consumer is built to degrade safely when
/// the set grows: it abstains from the optimisation (k-induction disables the
/// inductive step; GCSE havocs more).  Coarseness therefore costs
/// *optimisation opportunity*, never soundness.
///
/// The real risk is the opposite — **under**-approximation.  If the frontend
/// fails to model some way a pointer gets a value (external call, inline asm,
/// int->ptr cast, union type-pun, varargs), an empty set would let a consumer
/// wrongly conclude "nothing to invalidate" and apply an unsafe rewrite.
///
/// \ref TOP closes that hole.  It is the reserved node "may point anywhere".
/// Any assignment \ref collect_constraints cannot classify must route its LHS
/// to TOP (see \ref points_to_top).  The query layer then reports an
/// unnameable (`unknown`) target, which every consumer already treats as
/// "abstain / be maximally conservative".  Model-less code stays sound.
class andersent : public value_setst
{
public:
  andersent();

  /// A node is an abstract memory location: one program variable or one
  /// allocation site.  Nodes are dense integers so points-to sets can be
  /// cheap hashed-int sets (and, later, bitvectors).
  typedef unsigned node_id;

  /// Reserved node meaning "may point anywhere" (unknown / unmodelled target).
  /// Interned once by the constructor, so real program nodes start at 1 and
  /// \ref get_node never returns it.  See \ref andersen_soundness.
  static constexpr node_id TOP = 0;

  /// The four inclusion-constraint shapes of \ref andersen_constraints.
  enum class constraint_kindt
  {
    ADDRESS_OF, ///< `pts[lhs] ⊇ {rhs}`      (lhs = &rhs)
    COPY,       ///< `pts[lhs] ⊇ pts[rhs]`    (lhs = rhs)
    LOAD,       ///< `pts[lhs] ⊇ pts[*rhs]`   (lhs = *rhs)
    STORE       ///< `pts[*lhs] ⊇ pts[rhs]`   (*lhs = rhs)
  };

  struct constraintt
  {
    constraint_kindt kind;
    node_id lhs;
    node_id rhs;
  };

  /// \name Core solver (implement me)
  /// The pedagogical heart of the class.  These operate purely on node ids.
  /// @{

  /// Interns \p e as a node, returning a stable id (creates one on first use).
  /// The reverse mapping (node -> expr) powers the query layer.
  node_id get_node(const expr2tc &e);

  /// Registers one constraint.  Cheap; the real work happens in \ref solve.
  void add_constraint(constraint_kindt kind, node_id lhs, node_id rhs);

  /// Marks \p n as possibly pointing anywhere by adding \ref TOP to its set.
  /// The conservative escape hatch for anything the frontend cannot model.
  void points_to_top(node_id n);

  /// Computes the least fixpoint of all registered constraints, populating
  /// \ref pts.  **This is the exercise** — see \ref andersen_solve for the
  /// algorithm.  Idempotent: safe to call again after adding constraints.
  void solve();

  /// The points-to set of \p n after \ref solve (empty if never grown).
  const std::unordered_set<node_id> &points_to(node_id n) const;

  /// True iff \p a may point to \p b (i.e. `b ∈ pts[a]`).
  bool may_point_to(node_id a, node_id b) const;

  /// @}

  /// \name GOTO frontend (implement me, phase 2)
  /// @{

  /// Walks every function body and lowers pointer-relevant assignments to
  /// constraints via \ref add_constraint.  Left as a stub: fill in the
  /// irep2 pattern matching (symbol / address_of / dereference / member /
  /// index) as the second exercise.
  void collect_constraints(const goto_functionst &goto_functions);

  /// @}

  /// \name Query / transparency layer
  /// Lets whole-program consumers use Andersen wherever they used the
  /// value-set analysis.  Because the analysis is flow-insensitive, the
  /// location argument \p l is accepted for API compatibility and ignored.
  /// @{

  /// Runs the whole analysis: \ref collect_constraints then \ref solve.
  void operator()(const goto_functionst &goto_functions);

  /// \ref value_setst interface: values \p expr may hold, as a list of
  /// `object_descriptor2t` wrapping each pointee's program expression.
  void get_values(
    goto_programt::const_targett l,
    const expr2tc &expr,
    valuest &dest) override;

  /// The set of objects a dereference-style l-value can refer to.  Mirrors
  /// `value_sett::get_reference_set`, so the GCSE havoc path can switch to
  /// Andersen by calling this instead of reaching into `.value_set`.
  void get_reference_set(const expr2tc &expr, valuest &dest);

  /// Flow-insensitive: any location is "known" once the analysis has run.
  bool has_location(goto_programt::const_targett) const
  {
    return solved;
  }

  /// @}

protected:
  /// node -> its points-to set.  The analysis result.
  std::vector<std::unordered_set<node_id>> pts;

  /// All registered constraints (see \ref add_constraint).
  std::vector<constraintt> constraints;

  /// Copy edges `from -> to` built during \ref solve (dynamic edges from
  /// load/store are appended here as they are discovered).
  std::vector<std::vector<node_id>> copy_edges;

  /// expr <-> node bijection backing \ref get_node and the query layer.
  std::unordered_map<expr2tc, node_id, irep2_hash> expr_to_node;
  std::vector<expr2tc> node_to_expr;

  /// Set once \ref solve has produced a valid fixpoint.
  bool solved = false;

  /// Grows all node-indexed vectors so \p n is a valid index.
  void ensure_node(node_id n);

  /// Translates a solved points-to set into the `object_descriptor2t` list
  /// shape that \ref value_setst consumers expect.
  void to_object_descriptors(node_id n, valuest &dest) const;
};

#endif // CPROVER_POINTER_ANALYSIS_ANDERSEN_H
