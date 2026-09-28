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
/// a single **node** (see \ref andersent::node_id).  A heap allocation inside
/// a loop is one node no matter how many concrete objects it creates at run
/// time — that is the source of most of Andersen's imprecision, and it is what
/// keeps the analysis finite and fast.
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
/// | # | Statement | Constraint                   | Meaning    |
/// |---|-----------|------------------------------|------------|
/// | 1 | `p = &a;` | `pts[p] ⊇ {a}`               | address-of |
/// | 2 | `p = q;`  | `pts[p] ⊇ pts[q]`            | copy       |
/// | 3 | `p = *q;` | `∀o∈pts[q]. pts[p] ⊇ pts[o]` | load       |
/// | 4 | `*p = q;` | `∀o∈pts[p]. pts[o] ⊇ pts[q]` | store      |
///
/// Constraints 1 and 2 are *static*: they never change.  Constraints 3 and 4
/// are *dynamic*: the pairs they relate depend on the points-to sets, which
/// are still being computed — that is why \ref andersent::solve needs a
/// fixpoint (the classic O(n^3) dynamic transitive closure; cycle elimination
/// such as HCD/LCD is deliberately left out).
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
/// \ref andersent::TOP closes that hole.  It is the reserved node "may point
/// anywhere".  Any assignment \ref andersent::collect_constraints cannot
/// classify must route its LHS to TOP (see \ref andersent::points_to_top).
/// The query layer then reports an unnameable (`unknown`) target, which every
/// consumer already treats as "abstain / be maximally conservative".
/// Model-less code stays sound.
///
/// One statement cannot even be routed per object: inline asm, whose operands
/// IREP2 does not keep.  It sets \ref andersent::everything_escapes, and every
/// query then answers TOP.
class andersent : public value_setst
{
public:
  andersent();

  /// A node is an abstract memory location: one program variable or one
  /// allocation site.  Nodes are dense integers so points-to sets can be
  /// cheap hashed-int sets (and, later, bitvectors).
  typedef unsigned node_id;

  /// Reserved "may point anywhere"
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

  /// \name Core solver
  /// Operates purely on node ids, so it is exercised by the unit tests from
  /// synthetic constraints without needing a GOTO program.
  /// @{

  /// Interns \p e as a node, returning a stable id (creates one on first use).
  /// Symbols are interned by identifier alone: the same variable can reach the
  /// analysis with different-but-compatible types (a `symbol_type` at one use,
  /// the followed struct at another), and those must share a node.
  node_id get_node(const expr2tc &e);

  /// Registers one constraint.  Cheap; the real work happens in \ref solve.
  void add_constraint(constraint_kindt kind, node_id lhs, node_id rhs);

  /// Marks \p n as possibly pointing anywhere by adding \ref TOP to its set.
  /// The conservative escape hatch for anything the frontend cannot model.
  void points_to_top(node_id n);

  /// Computes the least fixpoint of all registered constraints, populating
  /// \ref pts.  Idempotent: safe to call again after adding constraints.
  void solve();

  /// The points-to set of \p n after \ref solve (empty if never grown).
  const std::unordered_set<node_id> &points_to(node_id n) const;

  /// True iff \p a may point to \p b (i.e. `b ∈ pts[a]`).
  bool may_point_to(node_id a, node_id b) const;

  /// @}

  /// \name GOTO frontend
  /// @{

  /// Walks every function body and lowers pointer-relevant assignments,
  /// returns and calls to constraints via \ref add_constraint.  Field-,
  /// index- and context-insensitive; see the implementation for the modelling
  /// choices.  Independent of \ref solve: it only builds constraints.
  ///
  /// Calls through a function pointer cannot be lowered here — their targets
  /// are exactly what is being computed — so they are recorded and left to
  /// \ref resolve_indirect_calls.  Use \ref operator(), which drives both.
  void collect_constraints(const goto_functionst &goto_functions);

  /// @}

  /// \name Query / transparency layer
  /// Lets whole-program consumers (GCSE, the k-induction pointer-array-write
  /// resolver) use Andersen wherever they used the value-set analysis.
  /// Because the analysis is flow-insensitive, the location argument \p l is
  /// accepted for API compatibility and ignored.
  /// @{

  /// Runs the whole analysis: \ref collect_constraints, then \ref solve and
  /// \ref resolve_indirect_calls alternately until a round adds nothing.
  void operator()(const goto_functionst &goto_functions);

  /// \ref value_setst interface: values \p expr may hold, as a list of
  /// `object_descriptor2t` wrapping each pointee's program expression.
  void get_values(locationt l, const expr2tc &expr, valuest &dest) override;

  /// The set of objects a dereference-style l-value can refer to.
  void
  get_reference_set(locationt l, const expr2tc &expr, valuest &dest) override;

  /// Flow-insensitive: any location is "known" once the analysis has run.
  bool has_location(locationt) const override
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
  /// load/store are appended here as they are discovered).  A set, because
  /// the load/store rules re-derive the same edge on every re-visit of a
  /// node and duplicates would otherwise accumulate without bound.
  std::vector<std::unordered_set<node_id>> copy_edges;

  /// Node tracking.  Symbols are keyed by identifier, everything else by the
  /// expression itself; see \ref get_node.
  std::unordered_map<irep_idt, node_id, irep_id_hash> symbol_to_node;
  std::unordered_map<expr2tc, node_id, irep2_hash> expr_to_node;
  std::vector<expr2tc> node_to_expr; // reverse lookup

  /// Counter behind \ref fresh_node.
  unsigned tmp_count = 0;

  /// Set once \ref solve has produced a valid fixpoint.
  bool solved = false;

  /// Set when the program contains a statement whose effect on memory is not
  /// visible in the IR at all (inline asm: IREP2 keeps only its source string,
  /// not the operands it writes through).  No per-object widening can be sound
  /// then, so every query answers TOP.
  bool everything_escapes = false;

  /// The node interned for \p e, or nullptr if it was never seen.
  const node_id *find_node(const expr2tc &e) const;

  /// Grows all node-indexed vectors so \p n is a valid index.
  void ensure_node(node_id n);

  /// A node with no program expression, used to hold the intermediate value
  /// of a compound right-hand side.  Never the target of an ADDRESS_OF, so it
  /// never appears inside a points-to set.
  node_id fresh_node();

  /// Unions `pts[src]` into `pts[dst]`; true iff `pts[dst]` grew.
  bool join(node_id dst, node_id src);

  /// Adds the copy edge `from -> to` and propagates along it immediately.
  /// True iff `pts[to]` grew.
  bool add_copy_edge(node_id from, node_id to);

  /// A node whose points-to set is `{TOP}`.  Used as the source of a STORE to
  /// express "an unmodelled callee may write anything through this pointer".
  node_id top_source();

  /// The node standing for \p fn's return value, shared by every call site
  /// (context insensitivity).
  node_id return_node(const irep_idt &fn);

  /// A node whose points-to set is the value of \p rhs, introducing a
  /// \ref fresh_node when the expression is not already a nameable object.
  node_id eval_rhs(const expr2tc &rhs, unsigned location_number);

  /// Lowers `lhs := rhs` to whichever of the four constraints fits.
  void handle_assign(
    const expr2tc &lhs,
    const expr2tc &rhs,
    unsigned location_number);

  /// Records that the object \p lhs designates may hold any address, storing
  /// through the pointer when \p lhs is a dereference.  The escape hatch for
  /// an assignment whose source this frontend cannot model.
  void assign_top(const expr2tc &lhs);

  /// Lowers an OTHER instruction: the ones that only read are dropped, the
  /// rest may write through any pointer operand.
  void handle_other(const expr2tc &code, unsigned location_number);

  /// Binds arguments to formals and the return value to the LHS.
  void handle_function_call(
    const code_function_call2t &call,
    const goto_functionst &goto_functions,
    unsigned location_number);

  /// A call through a function pointer.  Its targets are only known once the
  /// pointer's own set has been solved, so it is recorded here and lowered
  /// later by \ref resolve_indirect_calls.
  struct indirect_callt
  {
    node_id function; ///< node of the callee pointer
    expr2tc ret;
    std::vector<expr2tc> arguments;
    unsigned location_number;
    std::set<irep_idt> bound; ///< callees already lowered
    bool widened = false;     ///< conservative fallback already emitted
  };

  std::vector<indirect_callt> indirect_calls;

  /// Lowers every indirect call whose target set is now known.  True iff new
  /// constraints were added, meaning another \ref solve round is needed.
  bool resolve_indirect_calls(const goto_functionst &goto_functions);

  /// Binds \p arguments to \p callee's formals and its return value to \p ret.
  void bind_call(
    const irep_idt &callee_name,
    const goto_functiont &callee,
    const expr2tc &ret,
    const std::vector<expr2tc> &arguments,
    unsigned location_number);

  /// The conservative treatment of a callee this analysis cannot name: it may
  /// return anything and write anything through the pointers it is handed.
  void widen_call(
    const expr2tc &ret,
    const std::vector<expr2tc> &arguments,
    unsigned location_number);

  /// Translates a solved points-to set into the `object_descriptor2t` list
  /// shape that \ref value_setst consumers expect.
  void to_object_descriptors(node_id n, valuest &dest) const;
};

#endif // CPROVER_POINTER_ANALYSIS_ANDERSEN_H
