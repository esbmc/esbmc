/*
 * This function is used to check function contracts.
 * 
 * Verification Strategy: Abstraction and Hierarchical Verification
 * 
 * Function contracts enable a manual abstraction approach to assist over-approximation
 * verification logic. By splitting verification into system-level and function-level,
 * we should be able to reduce verification complexity. 
 * 
 * 1. Function-level verification (enforce_contracts):
 *    - Verify each function independently against its contract
 *    - Use contract as specification: assume requires -> execute function -> assert ensures
 *    - This provides over-approximation: if function satisfies contract, it's correct
 *    - Complexity: O(n) where n is function size, not system size
 * 
 * 2. System-level verification (replace_calls):
 *    - Replace function calls with contract semantics
 *    - Use contract as abstraction: assert requires -> havoc assigns -> assume ensures
 *    - This avoids exploring function body, reducing state space
 *    - Complexity: O(m) where m is call sites, not function implementations
 * 
 * Benefits:
 * - Modularity: Verify functions separately from system
 * - Scalability: System verification doesn't need to explore function internals
 * - Reusability: Once function is verified, contract can be used in any context
 * - Over-approximation: Contract provides safe abstraction (may have false positives)
 * 
 * This approach transforms a complex system verification problem into:
 * - Multiple simpler function verification problems
 * - One system verification problem using abstracted functions
 * 
 * It takes phases:
 * 1. Extract contract clauses (requires, ensures, assigns) from contract symbol
 * 2. For contract checking: rename original function, generate wrapper with assume requires -> call -> assert ensures
 * 3. For contract replacement: replace calls with assert requires -> havoc assigns -> assume ensures
 */

#ifndef ESBMC_CONTRACTS_H
#define ESBMC_CONTRACTS_H

#include <goto-programs/goto_functions.h>
#include <goto-programs/frame_enforcer.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>
#include <functional>
#include <map>
#include <set>
#include <string>

/// \brief Basic contract handling class
/// Provides contract checking and replacement functionality at goto function level
///
/// Function contracts are specifications that describe the behavior of functions.
/// They consist of:
/// - Preconditions (requires): conditions that must hold when the function is called
/// - Postconditions (ensures): conditions that must hold when the function returns
/// - Assigns clauses: memory locations that the function may modify
///
/// Contracts enable modular verification by allowing functions to be verified
/// independently and then used as abstractions in system-level verification.
class code_contractst
{
public:
  // ========== __ESBMC_is_fresh support for ensures ==========

  /// \brief Structure to store is_fresh mapping information
  struct is_fresh_mapping_t
  {
    irep_idt
      temp_var_name; ///< Temporary variable name (e.g., return_value$___ESBMC_is_fresh$1)
    expr2tc ptr_expr; ///< Pointer expression (dereferenced from &ptr)
    expr2tc size_expr; ///< Extent the contract asked for, in bytes; may be nil
  };

  code_contractst(
    goto_functionst &goto_functions,
    contextt &context,
    const namespacet &ns);

  /// \brief Enforce function contracts
  /// Renames function F to __ESBMC_contracts_original_F and generates a new wrapper function F
  /// Wrapper function: assume requires -> call original function -> assert ensures
  /// \param to_enforce Set of function names to enforce contracts for
  /// \param entry_function The --function entry point name (empty if using main).
  ///        When non-empty AND matches the function being enforced, the wrapper
  ///        allocates fresh backing storage for all pointer parameters so that
  ///        the harness-generated nil args become valid dereferenceable objects.
  /// \return The subset of \p to_enforce that was actually enforced. A name
  ///         absent from the result named nothing this pass could act on.
  std::set<std::string> enforce_contracts(
    const std::set<std::string> &to_enforce,
    const std::string &entry_function = "",
    bool check_assigns_compliance = false);

  /// \brief Replace function calls with contracts
  /// Replaces function calls with contract semantics:
  ///   1. Assert requires clause (check precondition)
  ///   2. Havoc all potentially modified locations:
  ///      - Assigns clause targets (if specified)
  ///      - Static lifetime global variables (conservative)
  ///      - Memory locations through pointer parameters
  ///   3. Assume ensures clause (assume postcondition)
  ///
  /// CRITICAL: We must havoc everything the function might modify,
  /// otherwise the effects cannot propagate from the removed function body.
  /// \param to_replace Set of function names to replace with contracts
  void replace_calls(const std::set<std::string> &to_replace);

  /// \brief Quick check if function has any contracts
  /// \param function_body Function goto program
  /// \return True if function has any contract clauses
  bool has_contracts(const goto_programt &function_body) const;

  /// \brief Check if function is marked with __attribute__((annotate("__ESBMC_contract")))
  /// \param func_sym Function symbol to check
  /// \return True if function has the contract annotation
  bool is_annotated_contract_function(const symbolt &func_sym) const;

  /// \brief Name of the first non-intrinsic call a clause in \p body depends
  ///   on, empty when there is none.
  std::string clause_call_callee(const goto_programt &body) const;

  /// \brief Diagnostic for such a call, empty when there is none.
  std::string clause_call_reason(const goto_programt &body) const;

  /// \brief Whether \p func_sym has a body carrying contract clauses, or the
  ///   __ESBMC_contract annotation. Used to pick the function the user
  ///   annotated when a short name matches symbols from several modes.
  bool declares_contracts(const symbolt &func_sym) const;

  /// \brief Code symbols whose short name is \p short_name and which satisfy
  ///   \p accept, in goto-function order.
  std::vector<symbolt *> short_name_candidates(
    const std::string &short_name,
    const std::function<bool(const symbolt &)> &accept);

  /// \brief Per-field snapshot for pointer-struct-field assigns compliance.
  /// Captures the pre-call value of a field NOT in the assigns clause so that
  /// the post-call assertion can verify it is unchanged.
  struct ptr_field_snapshot_t
  {
    expr2tc ptr_sym;      ///< Pointer symbol (e.g. symbol2tc for "ctx")
    type2tc pointee_type; ///< Resolved struct type pointed to by ptr_sym
    irep_idt field_name;  ///< Field name not in assigns (e.g. "capacity")
    type2tc field_type;   ///< Type of that field
    expr2tc snapshot_sym; ///< Snapshot symbol holding pre-call field value
  };

  /// \brief Snapshot for pointer-parameter dereference assigns compliance (Phase 2C).
  /// When a pointer param p is NOT declared in the assigns clause at all, its
  /// pointed-to value (*p or p->field for structs) must remain unchanged.
  struct ptr_deref_snapshot_t
  {
    expr2tc ptr_sym;      ///< Pointer parameter symbol
    type2tc pointee_type; ///< Resolved type pointed to by ptr_sym
    irep_idt field_name;  ///< Empty for scalars; field name for struct members
    type2tc value_type;   ///< Type of the snapshotted value
    expr2tc snapshot_sym; ///< Snapshot symbol holding the pre-call value
    expr2tc
      array_index; ///< Nil for scalars; nondet witness index for array fields
    type2tc
      member_type; ///< Array-field member type (for indexing); nil otherwise
    expr2tc alias_exemption; ///< Nil, or a guard under which this location is
                             ///< an assigns target reached by another name
  };

  /// \brief Snapshot for array element assigns compliance (Phase 2B).
  /// For __ESBMC_assigns(arr[declared_idx]), use a nondet witness index j
  /// to check that no other element arr[j] (j != declared_idx) was modified.
  struct arr_elem_snapshot_t
  {
    expr2tc arr_ptr;      ///< Array pointer symbol (e.g. symbol2tc for "arr")
    type2tc arr_add_type; ///< Result type of (arr + j) pointer-arithmetic
    type2tc elem_type;    ///< Element type (pointee of arr_ptr)
    expr2tc declared_idx; ///< Declared index expression (from assigns clause)
    expr2tc witness_idx;  ///< Nondet witness index symbol j
    expr2tc snapshot_sym; ///< Snapshot symbol holding arr[j] pre-call value
  };

  /// \brief Byte extent of a harness-allocated pointer parameter.
  ///
  /// \p justified says whether the harness backing is real enough to read
  /// through: an __ESBMC_is_fresh size, or the one-element stack backing of
  /// the #6483 carve-out. It is false for a nondet heap extent, which nothing
  /// may dereference. An absent map entry is a third state: the harness never
  /// allocated, so the pointer is the real caller's.
  struct param_extentt
  {
    expr2tc bytes;  ///< Byte-extent expression of the allocation
    bool justified; ///< True when the backing may be dereferenced
  };

  /// \brief Check if a function is compiler-generated and should be skipped.
  /// Handles both short names ("fst") and full Clang USR IDs ("c:@F@fst#*1I#").
  /// \param function_name Function name or full ID
  /// \return True if the function should be skipped (destructor, __cxa_*, etc.)
  bool is_compiler_generated(const std::string &function_name) const;

  /// \brief Say why \p function_name cannot be used by a contract flag.
  ///
  /// The eligibility rules differ between the two passes and always have:
  /// enforce_contracts resolves a name through find_function_symbol, while
  /// replace_calls selects through matches_replace_pattern, which is why a
  /// C++ id with parameters satisfies one and not the other. This method is
  /// the single place both rules live, so a caller can ask the same question
  /// the pass will ask, and get the same answer.
  ///
  /// Call it before any pass runs: enforce_contracts rewrites the functions it
  /// acts on into wrappers that carry no contract, so asking afterwards
  /// reports a name that was used as unusable.
  ///
  /// \param for_replace Ask replace_calls' rule rather than enforce's
  /// \return The reason, or an empty string when the name is usable
  std::string
  diagnose_contract_target(const std::string &function_name, bool for_replace);

private:
  goto_functionst &goto_functions;
  contextt &context;
  const namespacet &ns;
  frame_enforcert frame_enforcer;
  size_t ptr_field_snap_counter =
    0; ///< Counter for unique ptr-field snapshot names
  size_t ptr_deref_snap_counter =
    0; ///< Counter for unique ptr-deref snapshot names (Phase 2C)
  size_t arr_elem_snap_counter =
    0; ///< Counter for unique array-element snapshot names (Phase 2B)

  /// Fallback element count for the Phase 2B array-element witness index when
  /// no extent is recorded for the pointer: a global pointer, or a pointer
  /// parameter with no __ESBMC_is_fresh in a run without --function, where the
  /// entry harness never allocates. A global *array* does not reach here; it
  /// takes the whole-object snapshot path instead. It can over-bound the index
  /// and produce the spurious "array bounds violated" of #5314, so prefer a
  /// recorded extent whenever one exists.
  static constexpr size_t WITNESS_IDX_FALLBACK_ELEMS = 100;

  /// \brief Find function symbol
  /// \param function_name Function name (can be full ID or simple name)
  /// \return Pointer to function symbol, or nullptr if not found
  symbolt *find_function_symbol(const std::string &function_name);

  /// \brief Rename function
  /// \param old_id Original function ID
  /// \param new_id New function ID
  void rename_function(const irep_idt &old_id, const irep_idt &new_id);

  /// \brief Generate checking mode wrapper function body
  /// \param original_func Original function symbol
  /// \param requires_clause Requires expression
  /// \param ensures_clause Ensures expression
  /// \param original_func_id ID of the renamed original function
  /// \param original_body Original function body (before renaming)
  /// \param is_fresh_mappings Mappings for is_fresh temp variables in ensures
  /// \param alloc_ptr_params If true, allocate fresh malloc backing for all
  ///        pointer parameters (used in --function entry harness mode).
  /// \return Generated wrapper function body
  goto_programt generate_checking_wrapper(
    const symbolt &original_func,
    const expr2tc &requires_clause,
    const expr2tc &ensures_clause,
    const irep_idt &original_func_id,
    const goto_programt &original_body,
    const std::vector<is_fresh_mapping_t> &is_fresh_mappings,
    bool alloc_ptr_params = false,
    const std::vector<expr2tc> &assigns_targets = {},
    bool check_assigns_compliance = false);

  /// \brief A fresh lvalue symbol of \p type registered under \p name
  expr2tc
  declare_local_symbol(const std::string &name, const type2tc &type) const;

  /// \brief Declare and havoc the value a replaced call returns
  /// \param function_symbol Function symbol being called
  /// \param ret_val Place the call assigns to, nil when the result is dropped
  /// \param call_location Location to give the emitted instructions
  /// \param replacement Program the declaration and havoc are appended to
  /// \return The result symbol, nil for a function returning nothing
  expr2tc declare_call_result(
    const symbolt &function_symbol,
    const expr2tc &ret_val,
    const locationt &call_location,
    goto_programt &replacement) const;

  /// \brief Generate replacement code at function call site
  /// \param function_symbol Function symbol being called
  /// \param function_body Function body (to extract contracts from)
  /// \param call_instruction Function call instruction
  /// \param caller_body Function body containing the call
  void generate_replacement_at_call(
    const symbolt &function_symbol,
    const goto_programt &function_body,
    goto_programt::targett call_instruction,
    goto_programt &caller_body);

  /// \brief Extract requires clause from contract symbol
  /// \param contract_symbol Contract symbol
  /// \return Requires expression, or true_exprt() if not present
  expr2tc extract_requires_clause(const symbolt &contract_symbol);

  /// \brief Extract ensures clause from contract symbol
  /// \param contract_symbol Contract symbol
  /// \return Ensures expression, or true_exprt() if not present
  expr2tc extract_ensures_clause(const symbolt &contract_symbol);

  /// \brief Extract requires clauses from function body
  /// \param function_body Function goto program
  /// \return Requires expression (conjunction of all requires), or true_exprt() if none
  expr2tc extract_requires_from_body(const goto_programt &function_body);

  /// \brief Extract ensures clauses from function body
  /// \param function_body Function goto program
  /// \return Ensures expression (conjunction of all ensures), or true_exprt() if none
  expr2tc extract_ensures_from_body(const goto_programt &function_body);

  /// \brief Extract assigns clause from function body
  /// \param function_body Function goto program
  /// \return Vector of assign target expressions from __ESBMC_assigns()
  std::vector<expr2tc>
  extract_assigns_from_body(const goto_programt &function_body);

  /// \brief Extract assigns clause from contract symbol
  /// \param contract_symbol Contract symbol
  /// \return Assigns expression, or nil_exprt() if not present
  expr2tc extract_assigns_clause(const symbolt &contract_symbol);

  /// \brief Replace __ESBMC_return_value symbols in expression with actual return value
  /// \param expr Expression to replace symbols in
  /// \param ret_val Actual return value expression
  /// \return Expression with __ESBMC_return_value replaced
  expr2tc replace_return_value_in_expr(
    const expr2tc &expr,
    const expr2tc &ret_val) const;

  /// \brief Extract struct/union member accesses to temporary variables
  /// For struct return values, accessing members directly (ret_val.x) can cause
  /// symbolic execution issues when ret_val's value is a 'with' expression.
  /// This function extracts member accesses to temporary variables to avoid dereference failures.
  /// \param expr Expression containing member accesses
  /// \param ret_val Return value symbol (must be struct/union type)
  /// \param wrapper GOTO program to add temporary variable declarations and assignments
  /// \param location Source location for generated instructions
  /// \return Expression with member accesses replaced by temporary variables
  expr2tc extract_struct_members_to_temps(
    const expr2tc &expr,
    const expr2tc &ret_val,
    goto_programt &wrapper,
    const locationt &location);

  /// \brief Replace a symbol in expression with another expression
  /// \param expr Expression to replace symbols in
  /// \param old_symbol Symbol to replace
  /// \param new_expr Expression to replace with
  /// \return Expression with old_symbol replaced by new_expr
  expr2tc replace_symbol_in_expr(
    const expr2tc &expr,
    const expr2tc &old_symbol,
    const expr2tc &new_expr) const;

  /// \brief An assigns target with the callee's formals replaced by the
  ///        arguments of one call
  /// \param target_expr Assigns target as written in the callee
  /// \param function_symbol The callee
  /// \param actual_args Arguments at this call site
  /// \param[out] is_pointer_param Whether the target was a pointer parameter
  ///        and nothing else, the only shape whose havoc follows the pointer
  /// \return The target expressed in the caller's terms
  expr2tc instantiate_assigns_target(
    const expr2tc &target_expr,
    const symbolt &function_symbol,
    const std::vector<expr2tc> &actual_args,
    bool &is_pointer_param) const;

  // ========== __ESBMC_old support ==========

  /// \brief Structure to store old() snapshot information
  struct old_snapshot_t
  {
    expr2tc original_expr; ///< Expression inside __ESBMC_old()
    expr2tc snapshot_var;  ///< Snapshot variable symbol
  };

  /// \brief Check if expression is an __ESBMC_old() call
  /// \param expr Expression to check
  /// \return True if expr is a sideeffect with kind old_snapshot
  bool is_old_call(const expr2tc &expr) const;

  /// \brief Create a snapshot variable for an old() expression
  /// \param expr Expression to snapshot
  /// \param func_name Function name (for unique naming)
  /// \param index Index of this snapshot (for unique naming)
  /// \return Symbol expression for the snapshot variable
  expr2tc create_snapshot_variable(
    const expr2tc &expr,
    const std::string &func_name,
    size_t index) const;

  /// \brief Replace __ESBMC_old() calls with snapshot variables
  /// \param expr Expression containing old() calls
  /// \param snapshots Vector of snapshot information
  /// \return Expression with old() calls replaced by snapshot variables
  expr2tc replace_old_in_expr(
    const expr2tc &expr,
    const std::vector<old_snapshot_t> &snapshots) const;

  /// \brief Collect old_snapshot assignments from function body
  /// \param function_body GOTO program to scan for old_snapshot sideeffects
  /// \return Vector of old_snapshot_t structures (original_expr, temp_var)
  std::vector<old_snapshot_t>
  collect_old_snapshots_from_body(const goto_programt &function_body) const;

  /// \brief Snapshot fields of pointed-to structs that are NOT in the assigns clause.
  /// For each pointer symbol in classified.ptr_field_targets, enumerates the
  /// pointed-to struct's fields, and for each field NOT in the assigned set
  /// emits DECL+ASSIGN instructions capturing the pre-call value.
  /// \param classified Classified assigns targets (provides ptr_field_targets)
  /// \param original_func Original function symbol (provides parameter types)
  /// \param wrapper GOTO program to append snapshot instructions to
  /// \param location Source location for generated instructions
  /// \param func_name Function name for unique snapshot naming
  /// \return Vector of snapshot records for use in emit_ptr_field_assertions
  std::vector<ptr_field_snapshot_t> materialize_ptr_field_snapshots(
    const frame_enforcert::classified_assignst &classified,
    const symbolt &original_func,
    goto_programt &wrapper,
    const locationt &location,
    const std::string &func_name);

  /// \brief Emit ASSERT instructions checking that ptr->field is unchanged.
  /// For each snapshot in the vector, asserts ptr->field == snapshot_sym.
  /// \param snapshots Snapshots produced by materialize_ptr_field_snapshots
  /// \param wrapper GOTO program to append assertions to
  /// \param location Source location for generated instructions
  void emit_ptr_field_assertions(
    const std::vector<ptr_field_snapshot_t> &snapshots,
    goto_programt &wrapper,
    const locationt &location);

  // ========== Phase 2C: pointer-parameter dereference assigns compliance ==========

  /// \brief Snapshot pointer params whose dereferenced value is NOT in the assigns clause.
  /// For each pointer parameter p not covered by the assigns clause:
  ///   - scalar pointee: snapshot *p
  ///   - struct pointee: snapshot each field of *p
  /// Called before the function call in the checking wrapper.
  /// \param classified Classified assigns targets (provides pointer_targets, ptr_field_targets)
  /// \param assigns_targets Full assigns target list (must be non-empty to enable check)
  /// \param original_func Original function symbol (provides parameter types/names)
  /// \param wrapper GOTO program to append snapshot instructions to
  /// \param location Source location
  /// \param func_name Function name for unique snapshot naming
  /// \param param_extents Byte extent of each harness allocation. Params whose
  ///        backing is not justified are skipped: the snapshot dereferences
  ///        the pointer, and against a nondet extent that harness-invented
  ///        read fails its own bounds check, reporting a violation in a
  ///        parameter the contract never mentions.
  /// \return Vector of snapshot records for use in emit_ptr_deref_assertions
  std::vector<ptr_deref_snapshot_t> materialize_ptr_deref_snapshots(
    const frame_enforcert::classified_assignst &classified,
    const std::vector<expr2tc> &assigns_targets,
    const symbolt &original_func,
    goto_programt &wrapper,
    const locationt &location,
    const std::string &func_name,
    const std::map<irep_idt, param_extentt> &param_extents);

  /// \brief Snapshot one scalar element of an array field of *p.
  /// A whole-array rvalue read through the pointer is illegal C, so the
  /// element (*p).field[k] is captured at a nondet witness index k, clamped
  /// into range -- sound by the same forall-via-witness argument as Phase 2B.
  /// Appends to \p result unless the field is one this check skips (a VLA, a
  /// non-scalar element type, or a zero-length array, which has no element).
  /// \param deref_expr The *p expression the field is read from
  void materialize_ptr_deref_array_field(
    const irep_idt &param_id,
    const irep_idt &field,
    const type2tc &ftype,
    const type2tc &pointee,
    const expr2tc &ptr_sym,
    const expr2tc &deref_expr,
    goto_programt &wrapper,
    const locationt &location,
    const std::string &func_name,
    std::vector<ptr_deref_snapshot_t> &result);

  /// \brief Emit ASSERT instructions for pointer-parameter dereference compliance.
  /// For each snapshot: asserts *p == snapshot (scalar) or p->field == snapshot (struct).
  /// \param snapshots Snapshots produced by materialize_ptr_deref_snapshots
  /// \param wrapper GOTO program to append assertions to
  /// \param location Source location
  void emit_ptr_deref_assertions(
    const std::vector<ptr_deref_snapshot_t> &snapshots,
    goto_programt &wrapper,
    const locationt &location);

  // ========== Phase 2B: array element assigns compliance ==========

  /// \brief Materialize nondet witness snapshots for array element assigns compliance.
  /// For each dereference(add(arr, declared_idx)) in classified.pointer_targets:
  ///   - Creates a nondet witness index j (same type as declared_idx)
  ///   - Snapshots arr[j] before the function call
  /// \param classified Classified assigns targets (provides pointer_targets)
  /// \param assigns_targets Full assigns target list (must be non-empty to enable check)
  /// \param wrapper GOTO program to append snapshot instructions to
  /// \param location Source location
  /// \param func_name Function name for unique snapshot naming
  /// \param param_extents Byte extent of each harness allocation, used to
  ///        clamp the witness index to extent/sizeof(elem). An absent entry
  ///        falls back to WITNESS_IDX_FALLBACK_ELEMS. The bound is a clamp and
  ///        never an ASSUME: assuming a range that a zero or symbolic extent
  ///        can falsify would discharge the whole wrapper vacuously (#6212).
  /// \return Vector of snapshot records for use in emit_arr_elem_assertions
  std::vector<arr_elem_snapshot_t> materialize_arr_elem_snapshots(
    const frame_enforcert::classified_assignst &classified,
    const std::vector<expr2tc> &assigns_targets,
    goto_programt &wrapper,
    const locationt &location,
    const std::string &func_name,
    const std::map<irep_idt, param_extentt> &param_extents);

  /// \brief Emit ASSERT instructions for array element assigns compliance.
  /// For each snapshot: asserts (j == declared_idx) || (arr[j] == snapshot).
  /// \param snapshots Snapshots produced by materialize_arr_elem_snapshots
  /// \param wrapper GOTO program to append assertions to
  /// \param location Source location
  void emit_arr_elem_assertions(
    const std::vector<arr_elem_snapshot_t> &snapshots,
    goto_programt &wrapper,
    const locationt &location);

  /// \brief Materialize old snapshots in wrapper function (enforce-contract mode)
  /// Creates DECL and ASSIGN instructions for snapshot variables before function call
  /// \param old_snapshots Vector of snapshots to materialize (modified in-place)
  /// \param wrapper GOTO program to add snapshot instructions to
  /// \param func_name Function name for unique variable naming
  /// \param location Source location for generated instructions
  void materialize_old_snapshots_at_wrapper(
    std::vector<old_snapshot_t> &old_snapshots,
    goto_programt &wrapper,
    const std::string &func_name,
    const locationt &location) const;

  /// \brief Materialize old snapshots at call site (replace-call mode)
  /// Creates DECL and ASSIGN instructions for snapshot variables at call location
  /// \param old_snapshots Vector of snapshots from function body
  /// \param function_symbol Function symbol for parameter substitution
  /// \param actual_args Actual arguments at call site
  /// \param replacement GOTO program to add snapshot instructions to
  /// \param call_location Source location for generated instructions
  /// \return Vector of call-site snapshots (with parameter substitution applied)
  std::vector<old_snapshot_t> materialize_old_snapshots_at_callsite(
    const std::vector<old_snapshot_t> &old_snapshots,
    const symbolt &function_symbol,
    const std::vector<expr2tc> &actual_args,
    goto_programt &replacement,
    const locationt &call_location) const;

  // ========== Type fixing for return value comparisons ==========

  /// \brief Check if a symbol represents a return value variable
  /// \param sym Symbol to check
  /// \return True if symbol is a return value variable (matches patterns like "return_value", "__ESBMC_return_value", etc.)
  bool is_return_value_symbol(const symbol2t &sym) const;

  /// \brief Remove incorrect typecasts on return value symbols
  /// \param expr Expression to process
  /// \param ret_val Return value symbol with correct type
  /// \return Expression with incorrect casts removed
  expr2tc
  remove_incorrect_casts(const expr2tc &expr, const expr2tc &ret_val) const;

  /// \brief Fix type mismatches in comparison expressions involving return values
  /// \param expr Expression to fix (typically an ensures guard)
  /// \param ret_val Return value symbol with correct type
  /// \return Expression with corrected type casts
  expr2tc
  fix_comparison_types(const expr2tc &expr, const expr2tc &ret_val) const;

  /// \brief Normalize floating-point addition in contract expressions to use IEEE semantics
  /// This ensures contracts use IEEE_ADD (matching implementation) instead of regular +
  /// \param expr Expression to normalize (typically an ensures guard)
  /// \return Expression with floating-point add2t replaced by ieee_add2t
  expr2tc normalize_fp_add_in_ensures(const expr2tc &expr) const;

  /// \brief Normalize ensures guard expression for return value handling
  /// This is a unified helper that applies all return_value-related transformations:
  /// 1. Replaces __ESBMC_return_value with actual ret_val symbol
  /// 2. Fixes type mismatches in comparisons (removes incorrect casts, adds correct casts)
  /// 3. Normalizes floating-point operations to use IEEE semantics
  /// \param ensures_clause Original ensures clause expression
  /// \param ret_val Return value symbol (may be nil if function returns void)
  /// \return Normalized ensures guard ready for ASSERT/ASSUME
  expr2tc normalize_ensures_guard_for_return_value(
    const expr2tc &ensures_clause,
    const expr2tc &ret_val) const;

  // ========== __ESBMC_is_fresh support for ensures ==========

  /// \brief Extract is_fresh mappings from function body
  /// \param function_body Function goto program
  /// \return Vector of is_fresh mappings (temp var name -> pointer expr)
  std::vector<is_fresh_mapping_t>
  extract_is_fresh_mappings_from_body(const goto_programt &function_body) const;

  /// \brief Replace is_fresh temporary variables with a concrete predicate.
  /// \param expr Expression containing is_fresh temp variables
  /// \param mappings Vector of is_fresh mappings
  /// \param require_dynamic true for ensures (valid_object && is_dynamic);
  ///   false for a requires clause asserted at a --replace-call-with-contract
  ///   call site (valid_object only, so live stack/interior sub-objects are
  ///   accepted, #6380).
  /// \return Expression with is_fresh temp variables replaced
  expr2tc replace_is_fresh_temps(
    const expr2tc &expr,
    const std::vector<is_fresh_mapping_t> &mappings,
    bool require_dynamic) const;

  /// \brief Havoc assigns targets (similar to loop invariant approach)
  /// \param assigns_clause Assigns clause expression
  /// \param dest Destination goto program
  /// \param location Location information
  void havoc_assigns_targets(
    const expr2tc &assigns_clause,
    goto_programt &dest,
    const locationt &location);

  /// \brief Extract target variable list from assigns clause
  /// \param assigns_clause Assigns clause expression
  /// \return List of target variable expressions
  std::vector<expr2tc> extract_assigns_targets(const expr2tc &assigns_clause);

  /// \brief Havoc function parameters before checking the contract
  /// \param original_func Original function symbol
  /// \param dest Destination goto program (wrapper body)
  /// \param location Location information
  void havoc_function_parameters(
    const symbolt &original_func,
    goto_programt &dest,
    const locationt &location);

  /// \brief Havoc static lifetime global variables before checking the contract
  /// \param dest Destination goto program (wrapper body)
  /// \param location Location information
  void havoc_static_globals(goto_programt &dest, const locationt &location);

  /// \brief Allocate fresh malloc backing storage for all pointer parameters.
  /// Called in --function entry harness mode so that pointer params point to
  /// real heap objects instead of nil, enabling valid dereference in the body.
  ///
  /// The extent of each allocation is a fresh nondet value, so a parameter is
  /// only dereferenceable as far as the contract itself justifies via
  /// __ESBMC_is_fresh.  A fixed extent here would assume a buffer size the
  /// contract does not state and mask out-of-bounds accesses in the body
  /// (GitHub issue #6212). Struct and union params are the exception: they keep
  /// a one-element stack backing, see emit_struct_stack_backing.
  /// \param wrapper Destination goto program (wrapper body)
  /// \param func Function symbol
  /// \param location Location information
  /// \param skip_params Set of param IDs already allocated by __ESBMC_is_fresh
  /// \param separated_params Those of \p skip_params whose __ESBMC_is_fresh
  ///        the requires clause asserts unconditionally, and which therefore
  ///        state separation. Only these are withheld from aliasing.
  /// \param allocated_ptrs Output: snapshots of the heap allocations made
  ///        here, taken at allocation time by retain_allocation_for_free
  ///        rather than the lvalues themselves, which aliasing may reassign.
  ///        Stack-backed struct params are not appended. Callers use this to
  ///        emit matching free() calls at wrapper exit so --memory-leak-check
  ///        does not blame the user's function for wrapper-internal
  ///        allocations (CWE-401).
  /// \param param_extents Output: byte extent of each allocation, keyed by
  ///        parameter symbol, each tagged with whether it may be dereferenced.
  void add_pointer_validity_assumptions(
    goto_programt &wrapper,
    const symbolt &func,
    const locationt &location,
    const std::set<irep_idt> &skip_params,
    const std::set<irep_idt> &separated_params,
    std::vector<expr2tc> &allocated_ptrs,
    std::map<irep_idt, param_extentt> &param_extents);

  /// \brief Whether \p func can observe the extent of pointer parameter
  ///        \p param: dereferences it, or lets it escape into a call.
  ///
  /// Gates the unstated-extent warning so it is not raised for a parameter
  /// nothing reads through (#6511). Conservative: true whenever the body
  /// cannot be inspected, since a missed warning is worse than a spurious one.
  bool
  param_extent_is_observable(const symbolt &func, const irep_idt &param) const;

  /// \brief Lower __ESBMC_is_fresh in a requires clause for a replace site.
  ///
  /// \param separation Output: obligations the caller must discharge, one per
  ///        pair of arguments the contract declares separate.
  /// \return The requires clause with is_fresh temps rewritten.
  expr2tc lower_is_fresh_in_requires(
    const symbolt &function_symbol,
    const goto_programt &function_body,
    const std::vector<expr2tc> &actual_args,
    expr2tc requires_clause,
    std::vector<expr2tc> &separation);

  /// \brief Mark snapshots whose location an assigns target may also name.
  ///
  /// Pointer parameters may alias, so a parameter can be another name for
  /// memory the clause permits writing, and the frame assertion would report a
  /// violation that is not one. Matched on the field, so a sibling stays
  /// protected, and the base is read in the pre-state.
  void attach_alias_exemptions(
    std::vector<ptr_deref_snapshot_t> &result,
    const std::vector<expr2tc> &assigns_targets,
    const symbolt &original_func,
    goto_programt &wrapper,
    const locationt &location,
    const std::string &func_name);

  /// \brief Snapshot a just-made allocation so the wrapper can free it.
  ///
  /// The lvalue that received the allocation is not a reliable handle on it.
  /// Pointer parameters may alias (see emit_pointer_param_aliasing), so an
  /// lvalue can be reassigned, or two lvalues can turn out to be one and the
  /// second allocation overwrite the first. Freeing the lvalue would then free
  /// one object twice and leak the other. Freeing this snapshot cannot.
  ///
  /// \param name Distinguishes this snapshot's symbol within \p func.
  /// \return The snapshot symbol, to be registered for the matching free.
  expr2tc retain_allocation_for_free(
    goto_programt &wrapper,
    const expr2tc &allocated,
    const std::string &name,
    const symbolt &func,
    const locationt &location);

  /// \brief Let harness-backed pointer parameters alias one another.
  ///
  /// Backing each pointer parameter separately would let a callee's proof rest
  /// on the parameters addressing distinct objects, a hypothesis no contract
  /// clause states and nothing checks at a replace site. Enforcing a function
  /// and then replacing a call that passes one object twice would then prove
  /// properties false in the real program (issue #6551). Parameters covered by
  /// __ESBMC_is_fresh are excluded by their caller: is_fresh does state
  /// separation, so it keeps it.
  ///
  /// \param params Pointer parameters backed by the harness, each with the
  ///        pretty name used to build readable flag symbols.
  void emit_pointer_param_aliasing(
    goto_programt &wrapper,
    const symbolt &func,
    const locationt &location,
    const std::vector<std::pair<expr2tc, std::string>> &params);

  /// \brief Back a struct/union pointer param with one stack-allocated element.
  ///
  /// This is the normative statement of the #6483 carve-out; other sites point
  /// here rather than restating it. One element is still an extent the contract
  /// does not state (#6212), but the alternative is worse: a heap-backed struct
  /// silently discharges __ESBMC_old-based ensures clauses (#6483), turning
  /// every such contract into a false negative. Stack backing also gives symex
  /// proper SSA phi-nodes for conditional field writes, which the heap path
  /// loses. Route struct params through emit_pointer_param_malloc instead once
  /// #6483 is fixed.
  void emit_struct_stack_backing(
    goto_programt &wrapper,
    const expr2tc &p,
    const std::string &param_name,
    const type2tc &pointee,
    const symbolt &func,
    const locationt &location);

  /// \brief Emit malloc + non-null ASSUME for one pointer parameter.
  /// Allocates a nondet number of bytes and assigns the result to \p p. The
  /// caller registers a snapshot of \p p, not \p p itself, for the matching
  /// free: see retain_allocation_for_free for why the lvalue will not do.
  /// \return The byte-extent expression of the allocation.
  expr2tc emit_pointer_param_malloc(
    goto_programt &wrapper,
    const expr2tc &p,
    const std::string &param_name,
    const symbolt &func,
    const locationt &location);
};

#endif // ESBMC_CONTRACTS_H
