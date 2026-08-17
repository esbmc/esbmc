#include <cstdlib>
#include <algorithm>
#include <map>
#include <goto-programs/contracts/contracts.h>
#include <util/expr/type_byte_size.h>
#include <goto-programs/remove_no_op.h>
#include <util/expr/base_type.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/base/i2string.h>
#include <util/irep/std_expr.h>
#include <util/symtab/symbol.h>
#include <util/symtab/pretty.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/message/message.h>
#include <util/config/options.h>

/// Determine whether a contract clause instruction (ASSERT or ASSUME) should be
/// emitted for the given clause expression.
///
/// Returns false for trivially-redundant constant-bool clauses:
///   - ASSERT false_constant → always failing; keep it (returns true)
///   - ASSERT true_constant  → no-op assertion; skip it (returns false)
///   - ASSUME true_constant  → trivially satisfied; skip it (returns false)
///   - ASSUME false_constant → unsatisfiable; still meaningful; keep it (returns true)
///   - Non-constant          → always emit (returns true)
static bool should_add_clause_instruction(
  const expr2tc &clause,
  goto_program_instruction_typet inst_type)
{
  if (is_nil_expr(clause))
    return false;
  if (is_constant_bool2t(clause))
  {
    bool val = to_constant_bool2t(clause).value;
    return (inst_type == ASSERT) ? !val : val;
  }
  return true;
}

/// Apply a recursive transformation to all operands of an expression,
/// returning a (possibly modified) clone only when a change actually occurred.
/// This avoids an unnecessary clone when the transformation is a no-op.
///
/// \param expr  Source expression (not modified)
/// \param transform  Function applied to each direct sub-expression;
///                   if it returns the same pointer, no change is recorded
/// \return Either \p expr unchanged or a clone with updated operands
static expr2tc transform_operands_if_changed(
  const expr2tc &expr,
  const std::function<expr2tc(const expr2tc &)> &transform)
{
  expr2tc result = expr->clone();
  bool changed = false;
  result->Foreach_operand([&](expr2tc &op) {
    expr2tc new_op = transform(op);
    if (new_op != op)
    {
      op = new_op;
      changed = true;
    }
  });
  return changed ? result : expr;
}

/// Check if function name is __ESBMC_is_fresh (handles Clang USR format)
static bool is_fresh_function(const std::string &funcname)
{
  return funcname == "c:@F@is_fresh" || funcname == "__ESBMC_is_fresh" ||
         funcname.find("c:@F@__ESBMC_is_fresh") == 0 ||
         funcname.find("__ESBMC_is_fresh") == 0;
}

/// Whether \p e asserts the symbol \p name unconditionally: reachable from the
/// root through conjunctions, never under a disjunction, negation or
/// conditional.
///
/// A conditional `__ESBMC_requires(n <= 0 || __ESBMC_is_fresh(p, n))` claims
/// nothing about p on the other branch, so an obligation derived from it must
/// not be imposed on every caller.
static bool asserted_unconditionally(const expr2tc &e, const irep_idt &name)
{
  if (is_nil_expr(e))
    return false;

  if (is_and2t(e))
    return asserted_unconditionally(to_and2t(e).side_1, name) ||
           asserted_unconditionally(to_and2t(e).side_2, name);

  // A conjunct asserts the temp only by being it. Merely mentioning it, say
  // as an operand of a comparison, says nothing about whether the clause
  // demands it, so recursing into other node kinds would over-approximate.
  // The frontend may wrap the temp in a cast or compare it against zero.
  expr2tc leaf = e;
  while (is_typecast2t(leaf))
    leaf = to_typecast2t(leaf).from;
  // The frontend may compare the temp against zero with the constant on
  // either side; both are the same assertion.
  if (is_notequal2t(leaf))
  {
    const notequal2t &ne = to_notequal2t(leaf);
    if (is_constant_number(ne.side_2))
      leaf = ne.side_1;
    else if (is_constant_number(ne.side_1))
      leaf = ne.side_2;
  }
  while (is_typecast2t(leaf))
    leaf = to_typecast2t(leaf).from;

  return is_symbol2t(leaf) && to_symbol2t(leaf).thename == name;
}

/// Check if is_fresh call is in ensures clause by examining next instruction
static bool
is_fresh_in_ensures(goto_programt::const_targett it, const goto_programt &body)
{
  auto next_it = it;
  ++next_it;
  return next_it != body.instructions.end() && next_it->is_assume() &&
         id2string(next_it->location.comment()) == "contract::ensures";
}

/// Whether an __ESBMC_is_fresh call states separation, i.e. the requires
/// clause asserts it unconditionally. Under a guard the contract claims
/// nothing on the other branch, so the harness must not grant it there either.
static bool states_separation(
  const code_function_call2t &call,
  const expr2tc &requires_clause)
{
  return is_symbol2t(call.ret) &&
         asserted_unconditionally(
           requires_clause, to_symbol2t(call.ret).thename);
}

code_contractst::code_contractst(
  goto_functionst &_goto_functions,
  contextt &_context,
  const namespacet &_ns)
  : goto_functions(_goto_functions),
    context(_context),
    ns(_ns),
    frame_enforcer(_context)
{
}

bool code_contractst::is_compiler_generated(
  const std::string &function_name) const
{
  // Extract the short name from a Clang USR-style full ID.
  // C++ USR format: "c:@F@funcname#param_encoding#"
  // Strip everything from the first '#' to get "c:@F@funcname", then take
  // everything after the last '@' to get "funcname".
  // For plain short names (no '@', no '#') this is a no-op.
  std::string short_name = function_name;
  size_t hash_pos = short_name.find('#');
  if (hash_pos != std::string::npos)
    short_name.resize(hash_pos);
  size_t at_pos = short_name.rfind('@');
  if (at_pos != std::string::npos)
    short_name = short_name.substr(at_pos + 1);

  // Skip destructors
  if (!short_name.empty() && short_name[0] == '~')
    return true;

  // Skip C++ runtime helpers
  if (short_name.starts_with("__cxa_"))
    return true;

  // Skip already-processed contract wrappers
  if (function_name.find("__ESBMC_contracts_original_") == 0)
    return true;

  return false;
}

bool code_contractst::declares_contracts(const symbolt &func_sym) const
{
  // The annotation is on the symbol, so it holds whether or not a body is
  // available here; only the clause scan needs one.
  if (is_annotated_contract_function(func_sym))
    return true;

  auto it = goto_functions.function_map.find(func_sym.id);
  return it != goto_functions.function_map.end() && it->second.body_available &&
         has_contracts(it->second.body);
}

/// Code symbols in \p goto_functions whose short name is \p short_name and
/// which satisfy \p accept, in goto-function order.
std::vector<symbolt *> code_contractst::short_name_candidates(
  const std::string &short_name,
  const std::function<bool(const symbolt &)> &accept)
{
  std::vector<symbolt *> found;
  forall_goto_functions (it, goto_functions)
  {
    symbolt *candidate = context.find_symbol(it->first);
    if (
      candidate && candidate->get_type().is_code() &&
      id2string(candidate->name) == short_name && accept(*candidate))
      found.push_back(candidate);
  }
  return found;
}

symbolt *code_contractst::find_function_symbol(const std::string &function_name)
{
  // Exact match (handles full IDs like "c:@F@fst#*1I#" passed by wildcard
  // expansion)
  symbolt *sym = context.find_symbol(function_name);
  if (sym != nullptr)
    return sym;

  // A short name is resolved against every mode's symbols, and the operational
  // models define plenty of ordinary names. Whichever candidate actually
  // carries a contract is the one the user annotated, so prefer it over
  // whatever the naming conventions below happen to match first: a Python
  // `add` never has the id `c:@F@add`, but umath.c does.
  const std::vector<symbolt *> annotated = short_name_candidates(
    function_name, [this](const symbolt &s) { return declares_contracts(s); });

  if (annotated.size() == 1)
    return annotated.front();

  // Two annotated candidates share the name: skip the convention lookups so
  // the ambiguity is reported rather than silently resolved.
  if (annotated.empty())
  {
    // C convention: c:@F@funcname
    std::string func_id = "c:@F@" + function_name;
    sym = context.find_symbol(func_id);
    if (sym != nullptr)
      return sym;
    // C++ no-parameter free function: c:@F@funcname#
    sym = context.find_symbol(func_id + "#");
    if (sym != nullptr)
      return sym;
  }

  // C++ general fallback: search by short name to handle parameterized free
  // functions like c:@F@fst#*1I# where the user passes just "fst". Detect
  // ambiguity when multiple overloads share the same short name.
  const std::vector<symbolt *> all =
    short_name_candidates(function_name, [](const symbolt &) { return true; });

  if (all.size() > 1)
  {
    std::string ids;
    for (const symbolt *candidate : all)
      ids += (ids.empty() ? "" : ", ") + id2string(candidate->id);
    log_error(
      "Ambiguous function name '{}'; use a full symbol ID to disambiguate."
      " Candidates: {}",
      function_name,
      ids);
    return nullptr;
  }

  return all.empty() ? nullptr : all.front();
}

void code_contractst::rename_function(
  const irep_idt &old_id,
  const irep_idt &new_id)
{
  auto it = goto_functions.function_map.find(old_id);
  if (it == goto_functions.function_map.end())
  {
    log_error("Function {} not found for renaming", old_id);
    abort();
  }

  // Copy function to new name
  goto_functiont &old_func = it->second;
  goto_functions.function_map[new_id] = old_func;

  // Force-retag every instruction to the new function id (the default only
  // tags empty members, which would leave the copied body carrying the old id).
  // This matters for the entry point: when main is renamed to its
  // contracts_original copy, its END_FUNCTION must no longer be seen as main's
  // by symex — otherwise the per-main END_FUNCTION special-casing (assume(false)
  // to stop exploring post-main interleavings) fires inside the wrapper-called
  // original and kills the path before the wrapper's ensures assertion is ever
  // checked.
  goto_functions.function_map[new_id].update_instructions_function(
    new_id, /*force=*/true);

  // Update symbol table
  symbolt *old_sym = context.find_symbol(old_id);
  if (old_sym == nullptr)
  {
    log_error("Function symbol {} must exist in context", old_id);
    abort();
  }
  symbolt new_sym = *old_sym;
  new_sym.name = new_id;
  new_sym.id = new_id;
  context.add(new_sym);

  // Do NOT erase the old function yet - we'll replace it with the wrapper
}

// Forward declaration — defined after find_all_assignments below
static expr2tc inline_temporary_variables(
  const expr2tc &expr,
  const goto_programt &function_body,
  const goto_programt::const_targett &assume_location);

expr2tc
code_contractst::extract_requires_from_body(const goto_programt &function_body)
{
  std::vector<expr2tc> requires_clauses;

  // Scan function body for contract::requires annotations
  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_assume())
    {
      std::string comment = id2string(it->location.comment());
      if (comment == "contract::requires")
      {
        // Inline Clang temporaries and quantifier return values so the extracted
        // expression contains the actual forall/exists expression rather than a
        // dangling SSA symbol (e.g. return_value$___ESBMC_forall$1).
        expr2tc inlined_guard =
          inline_temporary_variables(it->guard, function_body, it);
        requires_clauses.push_back(inlined_guard);
      }
    }
  }

  // Combine all requires clauses with AND
  if (requires_clauses.empty())
    return gen_true_expr();
  if (requires_clauses.size() == 1)
    return requires_clauses[0];

  expr2tc result = requires_clauses[0];
  for (size_t i = 1; i < requires_clauses.size(); ++i)
  {
    result = and2tc(result, requires_clauses[i]);
  }
  return result;
}

// Helper structure to track assignments to a variable
/// Contract intrinsics keep their own materialisation in the wrapper, which
/// knows how to turn them into allocations, snapshots or quantifiers rather
/// than a plain re-issued call.
static bool is_contract_intrinsic(const std::string &funcname)
{
  if (is_fresh_function(funcname))
    return true;

  // Compare the base name rather than substring-matching the mangled id, so a
  // user function whose name merely contains one of these tokens is lifted
  // like any other.
  const size_t at = funcname.rfind('@');
  const std::string base =
    at == std::string::npos ? funcname : funcname.substr(at + 1);
  return base == "__ESBMC_old" || base == "__ESBMC_old_raw" ||
         base == "__ESBMC_forall" || base == "__ESBMC_exists";
}

static void collect_symbol_names(const expr2tc &e, std::set<irep_idt> &out)
{
  if (is_nil_expr(e))
    return;

  if (is_symbol2t(e))
  {
    out.insert(to_symbol2t(e).thename);
    return;
  }

  e->foreach_operand(
    [&out](const expr2tc &op) { collect_symbol_names(op, out); });
}

/// Calls in \p body whose results \p referenced depends on, transitively. A
/// clause call may take another call's result (`outer(inner(x))`), so the
/// operands of everything selected are themselves candidates; iterate until the
/// referenced set stops growing. \p referenced is grown in place.
static std::set<goto_programt::const_targett> select_clause_calls(
  std::set<irep_idt> &referenced,
  const std::set<irep_idt> &body_decls,
  const goto_programt &body)
{
  std::set<goto_programt::const_targett> selected;

  for (bool grew = true; grew;)
  {
    grew = false;
    forall_goto_program_instructions (it, body)
    {
      if (!it->is_function_call() || !is_code_function_call2t(it->code))
        continue;
      if (selected.count(it))
        continue;

      const code_function_call2t &call = to_code_function_call2t(it->code);
      if (is_nil_expr(call.ret) || !is_symbol2t(call.ret))
        continue;

      const irep_idt &ret_name = to_symbol2t(call.ret).thename;
      if (!referenced.count(ret_name) || !body_decls.count(ret_name))
        continue;

      // Contract intrinsics keep their own materialisation, which knows how to
      // turn them into allocations and snapshots rather than a plain call.
      if (
        is_symbol2t(call.function) &&
        is_contract_intrinsic(to_symbol2t(call.function).thename.as_string()))
        continue;

      selected.insert(it);
      for (const expr2tc &arg : call.operands)
        collect_symbol_names(arg, referenced);
      grew = true;
    }
  }

  return selected;
}

/// Symbols declared in \p body. A call can bind a global directly
/// (`g = compute();`); re-declaring that in the wrapper would shadow the very
/// object the clause is about, so only body-local temporaries are lifted.
static std::set<irep_idt> declared_in(const goto_programt &body)
{
  std::set<irep_idt> decls;
  forall_goto_program_instructions (it, body)
    if (it->is_decl() && is_code_decl2t(it->code))
      decls.insert(to_code_decl2t(it->code).value);
  return decls;
}

/// Name of the first non-intrinsic call whose result a clause in \p body
/// still depends on, empty when there is none.
///
/// A clause is lowered into a single ASSUME/ASSERT, so a call inside it
/// survives only as the SSA temporary the frontend bound its result to. That
/// temporary is declared in the function body, never in the wrapper or at a
/// replaced call site, so the clause is left over a free symbol: assumed it
/// constrains nothing, asserted it is unprovable (#6941). Contract intrinsics
/// are excluded because they carry their own materialisation.
std::string code_contractst::clause_call_callee(const goto_programt &body) const
{
  std::set<irep_idt> referenced;
  forall_goto_program_instructions (it, body)
    if (
      it->is_assume() &&
      (id2string(it->location.comment()) == "contract::requires" ||
       id2string(it->location.comment()) == "contract::ensures"))
      collect_symbol_names(it->guard, referenced);

  if (referenced.empty())
    return std::string();

  // The set is ordered by instruction address, not by position: operator< on a
  // const_targett is `&*i1 < &*i2` (goto_program.cpp) and instructiont lives in
  // a std::list, so its first element is whichever node the allocator placed
  // lowest. With two eligible calls that names an arbitrary one of them, and a
  // different one on a build whose heap is laid out differently. Walk the body
  // for the first call the program reaches instead.
  const std::set<goto_programt::const_targett> selected =
    select_clause_calls(referenced, declared_in(body), body);

  forall_goto_program_instructions (it, body)
    if (selected.count(it))
      return id2string(
        to_symbol2t(to_code_function_call2t(it->code).function).thename);

  return std::string();
}

std::string code_contractst::clause_call_reason(const goto_programt &body) const
{
  const std::string callee = clause_call_callee(body);
  if (callee.empty())
    return std::string();

  return "a contract clause calls '" + callee +
         "', whose result the clause cannot name outside the function body";
}

struct var_assignment_info
{
  expr2tc value;                         // The assigned value
  goto_programt::const_targett location; // Where the assignment is
  expr2tc condition; // Condition under which this assignment happens (if any)
};

// Helper function to find all assignments to a variable that can reach a given point
static std::vector<var_assignment_info> find_all_assignments(
  const irep_idt &var_name,
  const goto_programt &function_body,
  const goto_programt::const_targett &target_location)
{
  std::vector<var_assignment_info> assignments;

  // Search backwards from target_location to find all assignments
  // We stop at the beginning of function body or a point where the variable is declared
  goto_programt::const_targett search_it = target_location;
  while (search_it != function_body.instructions.begin())
  {
    --search_it;

    // Check if this is a DECL for our variable - stop searching
    if (search_it->is_decl() && is_code_decl2t(search_it->code))
    {
      const code_decl2t &decl = to_code_decl2t(search_it->code);
      if (decl.value == var_name)
        break;
    }

    // Check for assignment to our variable
    if (search_it->is_assign() && is_code_assign2t(search_it->code))
    {
      const code_assign2t &assign = to_code_assign2t(search_it->code);
      if (is_symbol2t(assign.target))
      {
        const symbol2t &target_sym = to_symbol2t(assign.target);
        if (target_sym.thename == var_name)
        {
          var_assignment_info info;
          info.value = assign.source;
          info.location = search_it;
          info.condition = expr2tc(); // Will be filled later if needed
          assignments.push_back(info);
        }
      }
    }
  }

  return assignments;
}

/// Reconstruct the value a temporary holds at \p use, by walking forward over
/// the region between its declaration and \p use and tracking the path
/// condition under which each assignment reaches that point.
///
/// Clang lowers a conditional whose arms contain calls -- in an ensures clause
/// that means __ESBMC_old -- into a temporary written on both arms:
///
///     ASSIGN v = NONDET
///     IF !G THEN GOTO L
///     ASSIGN v = <then>
///     GOTO M
///   L:
///     ASSIGN v = <else>
///   M: <use of v>
///
/// Picking one assignment loses the guard and the other arm, which yields an
/// ensures that is neither what was written nor a safe approximation of it: it
/// can be stronger (a false failure) or weaker (a false VERIFICATION
/// SUCCESSFUL). See #6499. Folding the reaching assignments back into an
/// if-then-else recovers the clause exactly.
///
/// \return The reconstructed value, or nil when the region is not a
///         forward-only DAG. Nil means "cannot reconstruct", never "no value":
///         callers must diagnose rather than fall back to a guess.
static expr2tc reconstruct_conditional_value(
  const irep_idt &var_name,
  const goto_programt &body,
  const goto_programt::const_targett &use)
{
  // Region start: the variable's declaration, or the start of the body.
  goto_programt::const_targett start = body.instructions.begin();
  for (goto_programt::const_targett it = use; it != body.instructions.begin();)
  {
    --it;
    if (it->is_decl() && is_code_decl2t(it->code))
      if (to_code_decl2t(it->code).value == var_name)
      {
        start = it;
        break;
      }
  }

  // Index the region so branch targets can be ordered against it.
  std::vector<goto_programt::const_targett> region;
  std::map<const goto_programt::instructiont *, size_t> index;
  for (goto_programt::const_targett it = start; it != use; ++it)
  {
    index[&*it] = region.size();
    region.push_back(it);
  }
  index[&*use] = region.size();
  region.push_back(use);

  // (path condition, value) pairs reaching each point. A nil value means the
  // variable is not yet assigned on that path. Paths carrying the same value
  // are merged on arrival: without that the list doubles at every branch, and
  // a clause with a handful of nested conditionals exhausts memory.
  std::vector<std::vector<std::pair<expr2tc, expr2tc>>> reaching(region.size());
  auto push = [&reaching](size_t at, const expr2tc &pc, const expr2tc &val) {
    for (auto &e : reaching[at])
      if (e.second == val)
      {
        e.first = or2tc(e.first, pc);
        return true;
      }
    // A clause needing more distinct values than this is beyond what a linear
    // fold should be trusted with; the caller diagnoses rather than guesses.
    if (reaching[at].size() >= 32)
      return false;
    reaching[at].emplace_back(pc, val);
    return true;
  };
  reaching[0].emplace_back(gen_true_expr(), expr2tc());

  for (size_t i = 0; i + 1 < region.size(); ++i)
  {
    if (reaching[i].empty())
      continue;

    goto_programt::const_targett it = region[i];

    // An assignment to the variable replaces the value on every path here.
    std::vector<std::pair<expr2tc, expr2tc>> out = reaching[i];
    if (it->is_assign() && is_code_assign2t(it->code))
    {
      const code_assign2t &a = to_code_assign2t(it->code);
      if (is_symbol2t(a.target) && to_symbol2t(a.target).thename == var_name)
      {
        // Expand the right-hand side once, here, so the two arms of the
        // if-then-else below share it. Leaving it to the caller re-expands
        // every nested temporary once per arm, and the clause grows
        // exponentially in its nesting depth.
        expr2tc v = inline_temporary_variables(a.source, body, it);
        for (auto &p : out)
          p.second = v;
      }
    }

    if (!it->is_goto())
    {
      for (auto &p : out)
        if (!push(i + 1, p.first, p.second))
          return expr2tc();
      continue;
    }

    // Only forward branches with a single in-region target are handled. A
    // backwards branch is a loop, which this linear fold cannot express.
    if (it->is_backwards_goto() || it->targets.size() != 1)
      return expr2tc();
    auto tgt = index.find(&*it->targets.front());
    if (tgt == index.end() || tgt->second <= i)
      return expr2tc();

    bool unconditional =
      is_constant_bool2t(it->guard) && to_constant_bool2t(it->guard).is_true();
    for (auto &p : out)
    {
      if (unconditional)
      {
        if (!push(tgt->second, p.first, p.second))
          return expr2tc();
        continue;
      }
      expr2tc g = inline_temporary_variables(it->guard, body, it);
      if (
        !push(tgt->second, and2tc(p.first, g), p.second) ||
        !push(i + 1, and2tc(p.first, not2tc(g)), p.second))
        return expr2tc();
    }
  }

  // Fold the paths that define the variable into nested if-then-elses.
  expr2tc result;
  for (const auto &p : reaching.back())
  {
    if (is_nil_expr(p.second))
      continue;
    expr2tc guard = p.first;
    simplify(guard);
    result = is_nil_expr(result)
               ? p.second
               : if2tc(p.second->type, guard, p.second, result);
  }
  return result;
}

// Helper function to inline temporary variables generated by Clang for short-circuit evaluation
// When ensures contains complex expressions like (a && b) || (c && d) with __ESBMC_old calls,
// Clang generates control flow with temporary variables (tmp$1, tmp$2, etc).
// This function recursively inlines these temporaries to get back the original expression.
//
// IMPORTANT: When a temporary variable has multiple assignments (conditional assignments),
// we need to handle this carefully. If one assignment is a constant (like 0 or 1) and the
// other is a more complex expression, we take the complex expression as the actual value.
static expr2tc inline_temporary_variables(
  const expr2tc &expr,
  const goto_programt &function_body,
  const goto_programt::const_targett &assume_location)
{
  if (is_nil_expr(expr))
    return expr;

  // If this is a symbol that looks like a Clang temporary (tmp$...), try to inline it
  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);
    std::string sym_name = id2string(sym.thename);

    // Check if this is a Clang-generated temporary variable or a quantifier
    // return value that needs to be inlined.
    // Matches:
    //   "$tmp"                     — Clang short-circuit temporaries
    //   "return_value$___ESBMC_forall$"  — result of __ESBMC_forall() call
    //   "return_value$___ESBMC_exists$"  — result of __ESBMC_exists() call
    // Note: We DON'T inline return_value$___ESBMC_old$X because those need to be
    // matched against snapshots by replace_old_in_expr
    bool is_clang_tmp = sym_name.find("$tmp") != std::string::npos;
    bool is_quantifier_retval =
      sym_name.find("return_value$___ESBMC_forall$") != std::string::npos ||
      sym_name.find("return_value$___ESBMC_exists$") != std::string::npos;
    if (
      (is_clang_tmp || is_quantifier_retval) &&
      sym_name.find("___ESBMC_old") == std::string::npos)
    {
      // Find all assignments to this variable
      std::vector<var_assignment_info> assignments =
        find_all_assignments(sym.thename, function_body, assume_location);

      if (assignments.empty())
      {
        // No assignment found, return as-is
        log_warning(
          "Could not find definition for temporary variable: {}", sym_name);
        return expr;
      }

      // If there's only one assignment, use it directly
      if (assignments.size() == 1)
      {
        const auto &assign = assignments[0];

        // Special case: if RHS is an old_snapshot sideeffect, DON'T inline further
        if (
          is_sideeffect2t(assign.value) &&
          to_sideeffect2t(assign.value).kind ==
            sideeffect2t::allockind::old_snapshot)
        {
          return expr;
        }

        // Recursively inline the RHS
        return inline_temporary_variables(
          assign.value, function_body, assign.location);
      }

      // Multiple assignments - this happens with conditional control flow
      // For short-circuit evaluation, Clang typically generates:
      //   if (!cond) goto fallback
      //   tmp = complex_expr
      //   goto end
      //   fallback: tmp = 0  (or 1)
      //   end: use tmp
      //
      // For OR expressions like (A && B) || (C && D), Clang generates:
      //   tmp$7 = NONDET      (initialization)
      //   tmp$7 = 1           (when first branch succeeds - short circuit)
      //   tmp$7 = tmp$6 ? 1:0 (when first branch fails, use second branch result)
      //
      // Fold the assignments back into an if-then-else under the guards that
      // select them, rather than choosing one and discarding the rest (#6499).
      expr2tc folded = reconstruct_conditional_value(
        sym.thename, function_body, assume_location);
      if (!is_nil_expr(folded))
        return folded;

      // Reconstruction failed, so any value chosen here would be a guess that
      // is neither the written clause nor a safe approximation of it.
      log_error(
        "cannot reconstruct the contract clause holding '{}': its control flow "
        "is not a forward-only region. Rewrite the clause without a "
        "conditional or short-circuit operator around __ESBMC_old.",
        sym_name);
      abort();

      expr2tc best_value;
      goto_programt::const_targett best_location =
        function_body.instructions.end();

      for (const auto &assign : assignments)
      {
        // Skip NONDET (initialization)
        if (
          is_sideeffect2t(assign.value) &&
          to_sideeffect2t(assign.value).kind == sideeffect2t::allockind::nondet)
        {
          continue;
        }

        // Skip trivial constant assignments (0, 1, false, true)
        // These are short-circuit markers, not the actual expression
        if (is_constant_bool2t(assign.value))
        {
          continue;
        }
        if (is_constant_int2t(assign.value))
        {
          auto val = to_constant_int2t(assign.value).value;
          if (val == 0 || val == 1)
            continue;
        }

        // Special case: if RHS is an old_snapshot sideeffect, DON'T inline further
        if (
          is_sideeffect2t(assign.value) &&
          to_sideeffect2t(assign.value).kind ==
            sideeffect2t::allockind::old_snapshot)
        {
          return expr;
        }

        // Found a meaningful assignment (references another temp or is a complex expr)
        best_value = assign.value;
        best_location = assign.location;
        break;
      }

      if (is_nil_expr(best_value))
      {
        // All assignments were trivial - this shouldn't happen for valid ensures
        // Fall back to returning the original expression
        log_warning(
          "Could not find meaningful assignment for temporary variable: {}",
          sym_name);
        return expr;
      }

      // Recursively inline the best value
      return inline_temporary_variables(
        best_value, function_body, best_location);
    }
  }

  // For all other expression types, recursively process operands
  expr2tc result = expr->clone();
  result->Foreach_operand([&](expr2tc &op) {
    op = inline_temporary_variables(op, function_body, assume_location);
  });

  return result;
}

expr2tc
code_contractst::extract_ensures_from_body(const goto_programt &function_body)
{
  std::vector<expr2tc> ensures_clauses;

  // Scan function body for contract::ensures annotations
  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_assume())
    {
      std::string comment = id2string(it->location.comment());
      if (comment == "contract::ensures")
      {
        // Inline any Clang-generated temporary variables to get the full expression
        // This handles cases where Clang generates control flow for short-circuit evaluation
        // The inline_temporary_variables function is smart enough to handle conditional
        // assignments by preferring non-trivial expressions over constant 0/1 values.
        expr2tc inlined_guard =
          inline_temporary_variables(it->guard, function_body, it);
        ensures_clauses.push_back(inlined_guard);
      }
    }
  }

  // Combine all ensures clauses with AND
  if (ensures_clauses.empty())
    return gen_true_expr();
  if (ensures_clauses.size() == 1)
    return ensures_clauses[0];

  expr2tc result = ensures_clauses[0];
  for (size_t i = 1; i < ensures_clauses.size(); ++i)
  {
    result = and2tc(result, ensures_clauses[i]);
  }
  return result;
}

expr2tc code_contractst::extract_requires_clause(const symbolt &contract_symbol)
{
  // Extract from contract symbol's value field
  // The value field should contain a struct with requires/ensures expressions
  if (contract_symbol.get_value().is_nil())
    return gen_true_expr();

  // For now, return the entire value as requires
  // TODO: Parse structured contract data if needed
  expr2tc req;
  migrate_symbol_value(contract_symbol, req);
  return req;
}

expr2tc code_contractst::extract_ensures_clause(const symbolt &contract_symbol)
{
  // Extract from contract symbol's value field
  if (contract_symbol.get_value().is_nil())
    return gen_true_expr();

  // TODO: Implement proper separation of requires and ensures from contract symbol
  // Currently, we extract ensures clauses from function body instead
  // This function is used for contract replacement mode which is not fully implemented yet
  log_warning(
    "extract_ensures_clause from contract symbol not fully implemented");
  return gen_true_expr();
}

// Helper function to check if function has an explicit empty assigns clause
// This distinguishes __ESBMC_assigns(0) from no assigns clause at all
static bool has_empty_assigns_marker(const goto_programt &function_body)
{
  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_assert())
    {
      std::string comment = id2string(it->location.comment());
      if (comment == "contract::assigns_empty")
      {
        return true;
      }
    }
  }
  return false;
}

// A frame condition is declared whether or not it names any target: an
// explicit __ESBMC_assigns() names nothing, which enforce_frame_rule reads as
// "every snapshotted global must be unchanged" -- the check that makes the
// empty clause mean anything (#6555).
static bool declares_frame_condition(
  const std::vector<expr2tc> &assigns_targets,
  const goto_programt &function_body)
{
  return !assigns_targets.empty() || has_empty_assigns_marker(function_body);
}

// Helper function to unwrap array-to-pointer decay in assigns targets
// In C, when an array is passed to a function, it decays to &arr[0].
// This function detects this pattern and returns the original array.
// Whether reaching \p e goes through a pointer, which is what decides whether
// an array-typed place can be written in one assignment.
static bool reaches_through_pointer(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (is_dereference2t(e))
    return true;

  bool found = false;
  e->foreach_operand([&found](const expr2tc &op) {
    if (reaches_through_pointer(op))
      found = true;
  });
  return found;
}

static expr2tc unwrap_array_decay(const expr2tc &expr)
{
  // Pattern: address_of(index(array, 0))
  if (is_address_of2t(expr))
  {
    const address_of2t &addr = to_address_of2t(expr);
    if (is_index2t(addr.ptr_obj))
    {
      const index2t &idx = to_index2t(addr.ptr_obj);
      // Any constant index, not only 0. `&b[2]` is not the decay a compiler
      // produces, but it names a place inside b that the callee then writes
      // forward from, and the array is the only object we can widen to. That
      // over-havocs b[0..1]; havocking more than the callee writes loses the
      // caller information rather than granting it any, so it is the safe
      // direction, and it is what a decayed `b` already does.
      if (is_constant_int2t(idx.index) && is_array_type(idx.source_value->type))
        return idx.source_value;
    }
  }

  return expr;
}

// The place a havoc writes to. Substituting a formal with an array argument
// re-introduces the decay the collection step stripped once — `p` becomes
// `&b[0]` — and an address is not a place to assign to. Widening back to the
// whole array also covers everything the callee reaches through the decayed
// pointer (#6961). Any other address_of names its own operand.
static expr2tc havoc_place(const expr2tc &target)
{
  expr2tc place = unwrap_array_decay(target);
  if (is_address_of2t(place))
    return to_address_of2t(place).ptr_obj;
  return place;
}

// Pointees a havoc has nothing to write through: void and function pointees
// name no object, and a pointer pointee is left alone under
// --add-symex-value-sets, as the loop-invariant havoc does. Shared by both
// pointer-havoc paths so the two cannot drift apart.
static bool skip_pointee_havoc(const type2tc &pointee)
{
  if (is_empty_type(pointee) || is_code_type(pointee) || is_nil_type(pointee))
    return true;

  return config.options.get_bool_option("add-symex-value-sets") &&
         is_pointer_type(pointee);
}

// A pointer parameter is pass-by-value, so the callee cannot change the
// caller's pointer, only what it points at. Havocking the argument itself both
// misses that write and invents a bogus pointer, which the ensures ASSUME then
// dereferences. Only the first element is reached this way; widening needs an
// object, which only the decay case names. Nil when there is nothing to write
// through.
static expr2tc havoc_through_pointer(const expr2tc &place, const namespacet &ns)
{
  if (!is_pointer_type(place))
    return place;

  type2tc pointee = ns.follow(to_pointer_type(place->type).subtype);
  if (skip_pointee_havoc(pointee))
    return expr2tc();

  return dereference2tc(pointee, place);
}

expr2tc code_contractst::instantiate_assigns_target(
  const expr2tc &target_expr,
  const symbolt &function_symbol,
  const std::vector<expr2tc> &actual_args,
  bool &is_pointer_param) const
{
  is_pointer_param = false;
  expr2tc instantiated = target_expr;

  if (!function_symbol.get_type().is_code())
    return instantiated;

  const code_typet::argumentst &params =
    to_code_type(function_symbol.get_type()).arguments();

  for (size_t i = 0; i < params.size() && i < actual_args.size(); ++i)
  {
    irep_idt param_id = params[i].get_identifier();
    if (param_id.empty() || is_nil_expr(actual_args[i]))
      continue;

    type2tc param_type = migrate_type(params[i].type());
    expr2tc param_symbol = symbol2tc(param_type, param_id);
    // The whole target is the formal, so the havoc has an argument to follow
    // rather than a subexpression that already names its own place.
    if (instantiated == param_symbol)
      is_pointer_param = is_pointer_type(param_type);
    instantiated =
      replace_symbol_in_expr(instantiated, param_symbol, actual_args[i]);
  }

  return instantiated;
}

std::vector<expr2tc>
code_contractst::extract_assigns_from_body(const goto_programt &function_body)
{
  std::vector<expr2tc> assigns_targets;

  log_debug(
    "contracts",
    "extract_assigns_from_body: scanning {} instructions",
    function_body.instructions.size());

  // Scan function body for assigns_target sideeffect assignments
  // These were created by __ESBMC_assigns() in builtin_functions.cpp
  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_assign())
    {
      const code_assign2t &assign = to_code_assign2t(it->code);

      // Check if RHS is a sideeffect with assigns_target
      if (
        is_sideeffect2t(assign.source) &&
        to_sideeffect2t(assign.source).kind ==
          sideeffect2t::allockind::assigns_target)
      {
        const sideeffect2t &se = to_sideeffect2t(assign.source);
        expr2tc target_expr = se.operand;

        // Unwrap array-to-pointer decay: &arr[0] -> arr
        // This happens when an array is passed to __ESBMC_assigns()
        target_expr = unwrap_array_decay(target_expr);

        // An array target travels as its address, so the marker instruction
        // does not read the array itself. Recover the place it names.
        if (
          is_address_of2t(target_expr) &&
          is_array_type(to_address_of2t(target_expr).ptr_obj->type))
          target_expr = to_address_of2t(target_expr).ptr_obj;

        log_debug("contracts", "  Found assigns target expression");
        assigns_targets.push_back(target_expr);
      }
    }
  }

  log_debug(
    "contracts",
    "extract_assigns_from_body: found {} assigns targets",
    assigns_targets.size());
  return assigns_targets;
}

expr2tc
code_contractst::extract_assigns_clause(const symbolt & /* contract_symbol */)
{
  // TODO: Extract assigns clause from contract symbol
  log_warning("extract_assigns_clause is not yet implemented");
  return expr2tc();
}

std::vector<expr2tc>
code_contractst::extract_assigns_targets(const expr2tc &assigns_clause)
{
  std::vector<expr2tc> targets;
  if (is_nil_expr(assigns_clause))
    return targets;

  // TODO: Parse assigns clause to extract target list
  targets.push_back(assigns_clause);
  return targets;
}

void code_contractst::havoc_assigns_targets(
  const expr2tc &assigns_clause,
  goto_programt &dest,
  const locationt &location)
{
  std::vector<expr2tc> targets = extract_assigns_targets(assigns_clause);
  if (targets.empty())
    return;

  for (const auto &target : targets)
  {
    expr2tc rhs = gen_nondet(target->type);
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->code = code_assign2tc(target, rhs);
    t->location = location;
    t->location.comment("contract assigns: assign non-deterministic value");
  }
}

void code_contractst::havoc_function_parameters(
  const symbolt &original_func,
  goto_programt &dest,
  const locationt &location)
{
  if (!original_func.get_type().is_code())
    return;

  const code_typet &code_type = to_code_type(original_func.get_type());
  const code_typet::argumentst &params = code_type.arguments();

  for (const auto &param : params)
  {
    // Build LHS symbol for the parameter
    type2tc param_type = migrate_type(param.type());
    expr2tc lhs = symbol2tc(param_type, param.get_identifier());

    // Do not assign nondeterministic values to pointers when value-set based
    // symex objects are enabled, to be consistent with loop invariant havoc.
    if (
      config.options.get_bool_option("add-symex-value-sets") &&
      is_pointer_type(lhs))
      continue;

    expr2tc rhs = gen_nondet(lhs->type);
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->code = code_assign2tc(lhs, rhs);
    t->location = location;
    t->location.comment("contract havoc parameter");
  }
}

// Python module-level globals carry static_lifetime=false so the C-side
// static-init pass does not const-propagate them (converter_stmt.cpp). They are
// still program-wide state a replaced call can write, and rw_set.cpp:180
// already recognises them by this rule.
static bool is_python_module_global(const symbolt &s)
{
  return s.mode == "Python" && !s.file_local;
}

void code_contractst::havoc_static_globals(
  goto_programt &dest,
  const locationt &location)
{
  // Iterate over all symbols in context to find static lifetime globals
  ns.get_context().foreach_operand([&dest, &location](const symbolt &s) {
    // Skip functions, types, and non-lvalue symbols
    if (s.get_type().is_code() || s.is_type || !s.lvalue)
      return;

    // Only process static lifetime variables (globals and static locals)
    if (!s.static_lifetime && !is_python_module_global(s))
      return;

    // Skip internal ESBMC symbols
    std::string sym_name = id2string(s.name);
    if (sym_name.starts_with("__ESBMC_"))
      return;

    // Build LHS symbol expression
    type2tc global_type = migrate_symbol_type(s);
    expr2tc lhs = symbol2tc(global_type, s.id);

    // Do not assign nondeterministic values to pointers when value-set based
    // symex objects are enabled, to be consistent with loop invariant havoc.
    if (
      config.options.get_bool_option("add-symex-value-sets") &&
      is_pointer_type(lhs))
      return;

    // Generate nondeterministic value and create assignment
    expr2tc rhs = gen_nondet(lhs->type);
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->code = code_assign2tc(lhs, rhs);
    t->location = location;
    t->location.comment("contract havoc global");
  });
}

std::set<std::string> code_contractst::enforce_contracts(
  const std::set<std::string> &to_enforce,
  const std::string &entry_function,
  bool check_assigns_compliance)
{
  std::set<std::string> enforced;
  for (const auto &function_name : to_enforce)
  {
    // Skip compiler-generated functions (destructors, constructors, exception handlers)
    // These functions are automatically generated by the compiler and should not have
    // user-defined contracts. Attempting to enforce contracts on them would be incorrect.
    if (is_compiler_generated(function_name))
    {
      continue;
    }

    symbolt *func_sym = find_function_symbol(function_name);
    if (func_sym == nullptr)
    {
      // Not necessarily absent: find_function_symbol also returns nullptr for
      // an ambiguous short name, and logs which candidates it found.
      log_warning("Could not resolve {} to a single function", function_name);
      continue;
    }

    // Find the function in goto_functions
    auto func_it = goto_functions.function_map.find(func_sym->id);
    if (
      func_it == goto_functions.function_map.end() ||
      !func_it->second.body_available)
    {
      log_warning("Function body for {} not available", function_name);
      continue;
    }

    // Quick check: skip if function has no contracts and no annotation
    // Functions can have contracts via:
    // 1. Explicit contract clauses in body (__ESBMC_requires, __ESBMC_ensures, __ESBMC_assigns)
    // 2. __attribute__((annotate("__ESBMC_contract"))) annotation (defaults to requires(true), ensures(true))
    bool has_explicit_contracts = has_contracts(func_it->second.body);
    bool has_annotation = is_annotated_contract_function(*func_sym);
    if (!has_explicit_contracts && !has_annotation)
    {
      continue;
    }

    // Save the original function body BEFORE renaming
    // Make a copy to avoid issues with iterator invalidation
    goto_programt original_body_copy = func_it->second.body;

    // Extract contract clauses from function body
    expr2tc requires_clause = extract_requires_from_body(original_body_copy);
    expr2tc ensures_clause = extract_ensures_from_body(original_body_copy);

    // Skip if no contracts found and no annotation
    // For annotated functions, we use default contracts (requires(true), ensures(true))
    // A contract exists if it's not a constant bool, or if it's a constant false
    // (gen_true_expr() is returned when no contract is found)
    bool has_requires = !is_constant_bool2t(requires_clause) ||
                        (is_constant_bool2t(requires_clause) &&
                         !to_constant_bool2t(requires_clause).value);
    bool has_ensures = !is_constant_bool2t(ensures_clause) ||
                       (is_constant_bool2t(ensures_clause) &&
                        !to_constant_bool2t(ensures_clause).value);

    // Extract assigns targets here so we can check them in the skip condition.
    // Functions with only __ESBMC_assigns (and no requires/ensures) still
    // deserve enforcement: the assigns compliance check is the contract.
    std::vector<expr2tc> assigns_targets_early =
      extract_assigns_from_body(original_body_copy);
    bool has_assigns =
      declares_frame_condition(assigns_targets_early, original_body_copy);

    // For annotated functions without explicit contracts, use default true/true
    // This allows the function to be processed with default contract semantics.
    // Also proceed if only an assigns clause is present (assigns-only contract).
    if (!has_requires && !has_ensures && !has_assigns && !has_annotation)
    {
      continue;
    }

    // CRITICAL: Always remove ensures ASSUME from renamed function
    // The ensures clause is checked in the wrapper function, not in the original function body.
    // Leaving ensures ASSUME in the original function would:
    // 1. Make the postcondition a precondition (assume before execution)
    // 2. Cause dereference failures for struct return values (accessing __ESBMC_return_value)
    // Therefore, we ALWAYS remove all contract::ensures assumptions.

    // Rename original function
    irep_idt original_id = func_sym->id;
    std::string original_name_str =
      "__ESBMC_contracts_original_" + function_name;
    irep_idt original_name_id(original_name_str);

    rename_function(original_id, original_name_id);

    // Remove requires/ensures ASSUMEs from renamed function.
    // The wrapper handles all contract enforcement; leaving them in contracts_original causes:
    // 1. Ensures: forcing postconditions as preconditions (wrong semantics)
    // 2. Requires with __ESBMC_old: dereference of void* value as pointer → alignment error
    // We need to properly update GOTO targets before removing instructions
    {
      auto &renamed_func = goto_functions.function_map[original_name_id];
      goto_programt &renamed_body = renamed_func.body;

      // Collect all contract requires/ensures instructions to remove
      std::set<goto_programt::targett> instructions_to_remove;
      for (auto it = renamed_body.instructions.begin();
           it != renamed_body.instructions.end();
           ++it)
      {
        if (it->is_assume())
        {
          std::string comment = id2string(it->location.comment());
          if (comment == "contract::ensures" || comment == "contract::requires")
          {
            instructions_to_remove.insert(it);

            // The GOTO pointer/arithmetic-safety pass emits auto-generated
            // ASSERTs (e.g. SAME-OBJECT for a pointer comparison) from the
            // clause expression, immediately preceding this ASSUME. They are
            // ASSERTs, not ASSUMEs, so the comment scan misses them; for an
            // ensures clause they reference __ESBMC_return_value, which is
            // unbound in the original body, yielding a spurious violation
            // (#5043). The wrapper re-checks the clause with return_value
            // properly bound, so drop the contiguous run of safety asserts
            // belonging to this clause. Contract clauses precede the function
            // body, so a real body assertion never sits directly before a
            // clause ASSUME; walking back over asserts cannot reach body code.
            for (auto p = it; p != renamed_body.instructions.begin();)
            {
              --p;
              if (!p->is_assert())
                break;
              instructions_to_remove.insert(p);
            }
          }
        }
      }

      // Build a map of instructions to remove -> their replacement target
      // The replacement target must NOT be an instruction that will also be removed
      typedef std::map<goto_programt::targett, goto_programt::targett>
        targets_mapt;
      targets_mapt targets_to_update;

      for (auto it : instructions_to_remove)
      {
        // Find the next instruction that will NOT be removed
        auto next_it = std::next(it);
        while (next_it != renamed_body.instructions.end() &&
               instructions_to_remove.count(next_it) > 0)
        {
          next_it = std::next(next_it);
        }
        targets_to_update[it] = next_it;
      }

      // Update all GOTO targets that point to removed instructions
      for (auto &inst : renamed_body.instructions)
      {
        if (inst.is_goto() || inst.is_catch())
        {
          for (auto &target : inst.targets)
          {
            auto map_it = targets_to_update.find(target);
            if (map_it != targets_to_update.end())
            {
              target = map_it->second;
            }
          }
        }
      }

      // Remove the instructions
      for (auto it : instructions_to_remove)
      {
        renamed_body.instructions.erase(it);
      }
    }

    // Recursive self-calls: under enforcement the function body is executed by
    // the checking wrapper, and a self-call resolves to that same wrapper, so
    // the real recursion is unwound unboundedly (OOM) instead of using the
    // function's own contract (#5313). Replace each self-call in the original
    // body with the contract (assert requires → havoc assigns → assume ensures),
    // exactly as --replace-call-with-contract does. original_body_copy still
    // carries the contract clauses (the renamed body has them stripped), so use
    // it as the contract source.
    {
      goto_programt &self_body =
        goto_functions.function_map[original_name_id].body;
      std::vector<goto_programt::targett> self_calls;
      Forall_goto_program_instructions (i_it, self_body)
      {
        if (!i_it->is_function_call() || !is_code_function_call2t(i_it->code))
          continue;
        const code_function_call2t &call = to_code_function_call2t(i_it->code);
        if (
          is_symbol2t(call.function) &&
          to_symbol2t(call.function).get_symbol_name() ==
            id2string(original_id))
          self_calls.push_back(i_it);
      }
      for (auto call_it : self_calls)
        generate_replacement_at_call(
          *func_sym, original_body_copy, call_it, self_body);
    }

    // Extract is_fresh mappings from function body for ensures clause replacement
    std::vector<code_contractst::is_fresh_mapping_t> is_fresh_mappings =
      extract_is_fresh_mappings_from_body(original_body_copy);

    // assigns_targets already extracted above (assigns_targets_early)
    // Reuse to avoid double scan.
    std::vector<expr2tc> assigns_targets = std::move(assigns_targets_early);

    // Allocate fresh malloc backing for pointer parameters when this function
    // is the --function entry point. In that mode ESBMC's harness passes nil
    // for all pointer args; without backing storage, dereferences in the body
    // or ensures clause would be invalid. When called from real code (no
    // entry_function, or a different function is the entry) the caller already
    // provides real pointers, so we must not overwrite them.
    //
    // NOTE: When --enforce-contract '*' is used, the wildcard expansion in
    // esbmc_parseoptions.cpp inserts full IDs like "c:@F@fst#*1I#" into
    // to_enforce, while --function gives only the short name "fst". Match
    // against both func_sym->name (short) and func_sym->id (full) to handle
    // both forms correctly.
    bool alloc_ptr_params =
      !entry_function.empty() && (id2string(func_sym->name) == entry_function ||
                                  id2string(func_sym->id) == entry_function);

    // Generate wrapper function, passing the original body
    goto_programt wrapper = generate_checking_wrapper(
      *func_sym,
      requires_clause,
      ensures_clause,
      original_name_id,
      original_body_copy,
      is_fresh_mappings,
      alloc_ptr_params,
      assigns_targets,
      check_assigns_compliance);

    // Create new function entry
    goto_functiont new_func;
    new_func.body = wrapper;
    if (func_sym->get_type().is_code())
      new_func.type = migrate_symbol_type(*func_sym);
    new_func.body_available = true;
    new_func.update_instructions_function(original_id);

    goto_functions.function_map[original_id] = new_func;

    log_status("Enforced contract for function {}", function_name);
    enforced.insert(function_name);
  }

  goto_functions.update();
  return enforced;
}

/// Build `malloc(size_bytes)` typed as u8*, so the allocation is exactly
/// size_bytes regardless of the pointee width.
///
/// The result is assigned to a `T *` lvalue without a cast. That is deliberate
/// and load-bearing: symex_assign dispatches on is_sideeffect2t(rhs) at the top
/// level, so wrapping the malloc in a typecast hides it and the assignment is
/// never routed to symex_mem. symex_mem inserts the cast itself
/// (memory_alloc.cpp: `if (rhs->type != lhs->type) rhs = typecast2tc(...)`).
static expr2tc byte_malloc(const expr2tc &size_bytes)
{
  type2tc char_type = get_uint8_type();
  return sideeffect2tc(
    pointer_type2tc(char_type),
    expr2tc(),
    size_bytes,
    std::vector<expr2tc>(),
    char_type,
    sideeffect2t::allockind::malloc);
}

goto_programt code_contractst::generate_checking_wrapper(
  const symbolt &original_func,
  const expr2tc &requires_clause,
  const expr2tc &ensures_clause,
  const irep_idt &original_func_id,
  const goto_programt &original_body,
  const std::vector<is_fresh_mapping_t> &is_fresh_mappings,
  bool alloc_ptr_params,
  const std::vector<expr2tc> &assigns_targets,
  bool check_assigns_compliance)
{
  goto_programt wrapper;
  locationt location = original_func.location;

  const bool declares_frame =
    declares_frame_condition(assigns_targets, original_body);

  // Note: Here is the design, enforce_contracts mode does NOT havoc
  // parameters or globals. The wrapper is called by actual callers, so we
  // preserve the caller's argument values. Global variables are handled by
  // unified nondet_static initialization, not per-function havoc.

  // 0. Process __ESBMC_is_fresh in requires: allocate memory FIRST.
  //    This must come before both pointer-validity assumptions and old-snapshot
  //    materialization so that:
  //      (a) valid_object(p) is true after the malloc, preventing vacuous
  //          ASSUME(false) from the entry-function pointer-validity path, and
  //      (b) old-snapshot assignments (e.g. __ESBMC_old(p->field)) can safely
  //          dereference pointers that the caller declared via __ESBMC_is_fresh.
  //    Ensures-clause is_fresh calls are handled separately via
  //    replace_is_fresh_in_ensures_expr and are excluded here.
  struct is_fresh_info
  {
    expr2tc ptr_arg;
    expr2tc size_expr;
    bool states_separation;
  };
  std::vector<is_fresh_info> is_fresh_calls;

  // Pointer-typed lvalues that received a heap allocation in this wrapper
  // (is_fresh requires-side mallocs and add_pointer_validity_assumptions
  // mallocs). Each gets a matching free() before the wrapper returns so that
  // --memory-leak-check does not blame the user's function for the
  // wrapper-internal allocation (CWE-401, see GitHub issue #4908).
  std::vector<expr2tc> wrapper_heap_ptrs;

  forall_goto_program_instructions (it, original_body)
  {
    if (it->is_function_call() && is_code_function_call2t(it->code))
    {
      const code_function_call2t &call = to_code_function_call2t(it->code);
      if (is_symbol2t(call.function))
      {
        std::string funcname = to_symbol2t(call.function).thename.as_string();
        if (
          is_fresh_function(funcname) &&
          !is_fresh_in_ensures(it, original_body) && call.operands.size() >= 2)
        {
          log_debug(
            "contracts",
            "is_fresh call found: funcname={}, noperands={}, "
            "op0_nil={}, op1_nil={}",
            funcname,
            call.operands.size(),
            is_nil_expr(call.operands[0]),
            is_nil_expr(call.operands[1]));
          is_fresh_info info;
          info.ptr_arg = call.operands[0]->clone();
          info.size_expr = call.operands[1]->clone();
          info.states_separation = states_separation(call, requires_clause);
          is_fresh_calls.push_back(info);
        }
      }
    }
  }

  // Byte extent of each harness-allocated pointer param, tagged with whether
  // the backing may be dereferenced. See param_extentt.
  //
  // Keyed by the pointer symbol's id. A consumer that looks up a different id
  // degrades silently to the WITNESS_IDX_FALLBACK_ELEMS path with no
  // diagnostic, so the two producers below must keep agreeing on it.
  std::map<irep_idt, param_extentt> param_extents;

  // is_fresh'd struct pointers, warned about only when the contract uses
  // __ESBMC_old at all (#6483).
  std::vector<std::string> is_fresh_struct_ptrs;

  // Sequence number for the retained-allocation symbols below. An is_fresh
  // lvalue may be indirect (a->p), so there is no parameter name to build a
  // unique symbol from.
  size_t is_fresh_alloc_seq = 0;

  // Emit the malloc + non-null assume for one resolved is_fresh pointer lvalue.
  auto emit_is_fresh_alloc =
    [&](const expr2tc &ptr_var, const expr2tc &size_expr) {
      goto_programt::targett assign_inst = wrapper.add_instruction(ASSIGN);
      assign_inst->code = code_assign2tc(ptr_var, byte_malloc(size_expr));
      assign_inst->location = location;
      assign_inst->location.comment("__ESBMC_is_fresh memory allocation");

      // Remember the allocation so the wrapper can free it before returning.
      // Freeing a snapshot rather than the lvalue itself matters when two
      // parameters alias: `a->p` and `b->p` are then the same lvalue, the
      // second malloc overwrites the first, and freeing the lvalue twice is a
      // double free of one object while the other leaks.
      wrapper_heap_ptrs.push_back(retain_allocation_for_free(
        wrapper,
        ptr_var,
        "isfresh" + std::to_string(is_fresh_alloc_seq++),
        original_func,
        location));

      if (is_symbol2t(ptr_var) && is_pointer_type(ptr_var->type))
      {
        // The contract asked for this allocation, so its extent is justified.
        param_extents[to_symbol2t(ptr_var).thename] = {size_expr, true};

        // A struct/union pointee bypasses the stack-backing carve-out and gets
        // the heap object #6483 makes unsound. Only an __ESBMC_old over that
        // pointer can trip it, so stay quiet otherwise rather than training
        // users to ignore the warning.
        if (is_structure_type(
              ns.follow(to_pointer_type(ptr_var->type).subtype)))
          is_fresh_struct_ptrs.push_back(
            get_pretty_name(id2string(to_symbol2t(ptr_var).thename)));
      }

      // Assume the pointer is non-null: __ESBMC_is_fresh guarantees a fresh,
      // valid memory block.  Without this, symex_mem's non-deterministic
      // malloc-failure path can produce NULL, causing later derefs to fail.
      auto assume_nn = wrapper.add_instruction(ASSUME);
      assume_nn->guard = notequal2tc(ptr_var, gen_zero(ptr_var->type));
      assume_nn->location = location;
      assume_nn->location.comment("__ESBMC_is_fresh: pointer is non-null");
    };

  // is_fresh pointers whose lvalue reads through another pointer — e.g. a
  // member access `this->p` in a C++ method — must be allocated AFTER that base
  // pointer's harness storage is set up. add_pointer_validity_assumptions (step
  // 1) havocs the receiver struct with NONDET, which would otherwise clobber a
  // malloc emitted here, leaving `this->p` pointing at a nondet (invalid)
  // address. Defer those to step 1b. Plain symbol params are not havoced in
  // enforce mode, so allocate them up-front where old-snapshots and validity
  // assumptions can see them. Issue #6.
  std::vector<is_fresh_info> deferred_is_fresh;

  for (const auto &info : is_fresh_calls)
  {
    // Strip typecasts (e.g. (void*)(&hdr_len) → &hdr_len) to recover the
    // true type.  __ESBMC_is_fresh takes void* so the frontend inserts an
    // implicit cast, losing the actual type of the pointed-to pointer.
    expr2tc stripped = info.ptr_arg;
    while (is_typecast2t(stripped))
      stripped = to_typecast2t(stripped).from;

    // Determine the lvalue to assign the malloc result to.
    //
    // __ESBMC_is_fresh(EXPR, n) means "EXPR points to a fresh n-byte
    // allocation".  The malloc result therefore goes INTO EXPR — EXPR is
    // already the pointer-typed lvalue we want to write.  Two equivalent
    // surface forms appear in practice:
    //
    //   __ESBMC_is_fresh(p,  n)   — bare pointer (the canonical form per
    //                                docs/function-contracts.md);
    //                                stripped is the pointer expression p.
    //   __ESBMC_is_fresh(&p, n)   — address-of-variable form;
    //                                stripped is &p, so the lvalue is *(&p) = p.
    //
    // Both forms must end up assigning the malloc result to p itself, never
    // to *p.  Earlier revisions had a "fallback" else branch that emitted
    // *p = malloc(...), which dereferenced an uninitialised p and tripped
    // alignment / pointer-validity checks before the body even ran.
    // Peeling the address-of is only sound when what it wraps is itself a
    // pointer.  For &obj with obj a struct or scalar, the peel would assign
    // the malloc result into obj: value-set analysis then walks a pointer as
    // if it were a struct and aborts (#6469), and for a scalar obj the
    // assignment silently clobbers it and the contract means nothing.
    // An address_of is itself pointer-typed, so it has to be matched before
    // the bare-pointer case or it falls through to an unassignable lvalue.
    expr2tc ptr_var;
    bool assignable = true;
    if (is_address_of2t(stripped))
    {
      const expr2tc &obj = to_address_of2t(stripped).ptr_obj;
      assignable = is_pointer_type(obj->type);
      ptr_var = obj;
    }
    else
    {
      assignable = is_pointer_type(stripped->type);
      ptr_var = stripped;
    }

    if (!assignable)
    {
      log_error(
        "__ESBMC_is_fresh needs a pointer it can point at fresh storage, but "
        "the contract of '{}' gives it the address of a non-pointer object. "
        "Take that object as a pointer parameter, or allocate it with malloc "
        "and pass the pointer.",
        id2string(original_func.name));
      abort();
    }

    // A plain symbol lvalue is independent of any harness setup — allocate now.
    // An indirect lvalue (member/index/dereference) reads through a base
    // pointer that step 1 sets up, so defer it.
    if (is_symbol2t(ptr_var))
      emit_is_fresh_alloc(ptr_var, info.size_expr);
    else
    {
      is_fresh_info deferred;
      deferred.ptr_arg = ptr_var;
      deferred.size_expr = info.size_expr;
      deferred_is_fresh.push_back(deferred);
    }
  }

  // Collect the set of params already allocated by __ESBMC_is_fresh so that
  // add_pointer_validity_assumptions can skip them (avoids overwriting a
  // correctly-sized is_fresh allocation with a single-element malloc).
  // Both surface forms must be recognised: __ESBMC_is_fresh(&p, n) yields
  // an address_of2t over symbol p; __ESBMC_is_fresh(p, n) yields the
  // bare symbol expression p.
  //
  // Allocation and separation are tracked apart. A guarded is_fresh still owns
  // its allocation, so it must be skipped here too, but it states no
  // separation, so it stays aliasable: otherwise the harness grants a callee
  // separation on a branch the contract says nothing about, and the replace
  // side -- which gates its obligation on the same test -- asks the caller for
  // nothing. That is the composition hole this pass exists to close, reopened.
  std::set<irep_idt> is_fresh_allocated_params, is_fresh_separated_params;
  for (const auto &info : is_fresh_calls)
  {
    expr2tc stripped = info.ptr_arg;
    while (is_typecast2t(stripped))
      stripped = to_typecast2t(stripped).from;
    if (is_address_of2t(stripped))
      stripped = to_address_of2t(stripped).ptr_obj;
    if (!is_symbol2t(stripped))
      continue;

    const irep_idt name = to_symbol2t(stripped).thename;
    is_fresh_allocated_params.insert(name);
    if (info.states_separation)
      is_fresh_separated_params.insert(name);
  }

  // 1. Allocate fresh backing storage for pointer parameters (entry-function
  //    harness mode only). Comes after is_fresh allocation so that pointers
  //    declared via __ESBMC_is_fresh already have valid storage and are skipped.
  if (alloc_ptr_params)
  {
    add_pointer_validity_assumptions(
      wrapper,
      original_func,
      location,
      is_fresh_allocated_params,
      is_fresh_separated_params,
      wrapper_heap_ptrs,
      param_extents);
  }

  // 1b. Allocate deferred is_fresh pointers (those reading through a base
  //     pointer, e.g. a C++ method's this->member) now that the base pointer's
  //     harness storage exists and has been havoced. See the note where
  //     deferred_is_fresh is populated. Issue #6.
  for (const auto &info : deferred_is_fresh)
    emit_is_fresh_alloc(info.ptr_arg, info.size_expr);

  // 2. Extract and create snapshots for __ESBMC_old() expressions.
  //    Comes after is_fresh allocation so that old-snapshot assignments can
  //    safely dereference pointers that were set up above.
  std::vector<old_snapshot_t> old_snapshots =
    collect_old_snapshots_from_body(original_body);

  if (!old_snapshots.empty() && !is_fresh_struct_ptrs.empty())
    log_warning(
      "{}: __ESBMC_is_fresh on struct pointer(s) {} heap-backs them, which can "
      "silently discharge __ESBMC_old-based ensures clauses (#6483).",
      location,
      fmt::join(is_fresh_struct_ptrs, ", "));

  materialize_old_snapshots_at_wrapper(
    old_snapshots, wrapper, id2string(original_func.name), location);

  // Lambda function to add contract clause instruction (ASSERT or ASSUME)
  // Used for both requires (ASSUME) and ensures (ASSERT) clauses in enforce mode
  auto add_contract_clause = [&wrapper, &location](
                               const expr2tc &clause,
                               const goto_program_instruction_typet inst_type,
                               const std::string &comment) {
    if (!should_add_clause_instruction(clause, inst_type))
      return;

    goto_programt::targett t = wrapper.add_instruction(inst_type);
    t->guard = clause;
    t->location = location;
    t->location.comment(comment);
  };

  // 3. Assume requires clause (after memory allocation for is_fresh)
  // Also replace __ESBMC_old() references in the requires clause — old snapshots
  // have already been materialized above (step 2), so replacement is safe here.
  {
    expr2tc req = requires_clause;
    if (!old_snapshots.empty() && !is_nil_expr(req))
      req = replace_old_in_expr(req, old_snapshots);
    add_contract_clause(req, ASSUME, "contract requires");
  }

  // 3b. Snapshot globals and ptr->field targets for assigns compliance
  //     (before function call).
  // Only runs when assigns clause is explicitly declared: without an assigns
  // clause the function may modify anything, so there is nothing to check.
  std::vector<ptr_field_snapshot_t> ptr_field_snaps;
  std::vector<ptr_deref_snapshot_t> ptr_deref_snaps;
  std::vector<arr_elem_snapshot_t> arr_elem_snaps;
  frame_enforcert::classified_assignst classified_assigns;
  if (check_assigns_compliance && declares_frame)
  {
    std::string func_name = id2string(original_func.name);

    // 3b-i. Snapshot global variables
    auto globals = frame_enforcert::collect_global_variables(context);
    if (!globals.empty())
    {
      log_debug(
        "contracts",
        "generate_checking_wrapper: snapshotting {} globals for assigns "
        "compliance of {}",
        globals.size(),
        func_name);
      frame_enforcer.materialize_snapshots(
        globals, wrapper, location, "contract_" + func_name);
    }

    // 3b-ii. Classify assigns targets (needed for both ptr-field and ptr-deref)
    classified_assigns =
      frame_enforcert::classify_assigns_targets(assigns_targets);

    // 3b-iii. Snapshot ptr->field targets (for local pointer parameter fields)
    if (!classified_assigns.ptr_field_targets.empty())
    {
      ptr_field_snaps = materialize_ptr_field_snapshots(
        classified_assigns, original_func, wrapper, location, func_name);
      log_debug(
        "contracts",
        "generate_checking_wrapper: created {} ptr-field snapshots for {}",
        ptr_field_snaps.size(),
        func_name);
    }

    // 3b-iv. Phase 2C: snapshot *p for pointer params NOT in assigns at all
    ptr_deref_snaps = materialize_ptr_deref_snapshots(
      classified_assigns,
      assigns_targets,
      original_func,
      wrapper,
      location,
      func_name,
      param_extents);
    if (!ptr_deref_snaps.empty())
    {
      log_debug(
        "contracts",
        "generate_checking_wrapper: created {} ptr-deref snapshots for {} "
        "(Phase 2C)",
        ptr_deref_snaps.size(),
        func_name);
    }

    // 3b-v. Phase 2B: nondet-witness snapshot for array element assigns
    arr_elem_snaps = materialize_arr_elem_snapshots(
      classified_assigns,
      assigns_targets,
      wrapper,
      location,
      func_name,
      param_extents);
    if (!arr_elem_snaps.empty())
    {
      log_debug(
        "contracts",
        "generate_checking_wrapper: created {} array-elem snapshots for {} "
        "(Phase 2B)",
        arr_elem_snaps.size(),
        func_name);
    }
  }

  // 2. Declare return value variable (if function has return type)
  expr2tc ret_val;
  type2tc ret_type;
  if (original_func.get_type().is_code())
  {
    const code_typet &code_type = to_code_type(original_func.get_type());
    typet return_type_irep1 = code_type.return_type();
    log_debug(
      "contracts",
      "generate_checking_wrapper: original return_type (irep1) id={}, "
      "identifier={}",
      return_type_irep1.id().as_string(),
      return_type_irep1.id() == "symbol"
        ? return_type_irep1.identifier().as_string()
        : "N/A");

    // Resolve symbol_type to concrete type using ns.follow()
    // This is critical: value set analysis cannot handle symbol_type
    if (return_type_irep1.id() == "symbol")
    {
      log_debug(
        "contracts",
        "generate_checking_wrapper: resolving symbol_type {}",
        return_type_irep1.identifier().as_string());
      return_type_irep1 = ns.follow(return_type_irep1);
      log_debug(
        "contracts",
        "generate_checking_wrapper: resolved to type id={}",
        return_type_irep1.id().as_string());
    }

    ret_type = migrate_type(return_type_irep1);
    log_debug(
      "contracts",
      "generate_checking_wrapper: ret_type (irep2) type_id={}, "
      "is_symbol_type={}",
      ret_type ? get_type_id(*ret_type) : "nil",
      ret_type && is_symbol_type(ret_type));

    // Also resolve symbol_type2t in irep2 if needed
    if (is_symbol_type(ret_type))
    {
      log_debug(
        "contracts",
        "generate_checking_wrapper: ret_type is symbol_type2t, resolving...");
      ret_type = ns.follow(ret_type);
      log_debug(
        "contracts",
        "generate_checking_wrapper: resolved ret_type type_id={}",
        ret_type ? get_type_id(*ret_type) : "nil");
    }

    if (!is_nil_type(ret_type))
    {
      // Create and add symbol to symbol table
      irep_idt ret_val_id("__ESBMC_return_value");
      symbolt ret_val_symbol;
      ret_val_symbol.name = ret_val_id;
      ret_val_symbol.id = ret_val_id;
      ret_val_symbol.set_type(return_type_irep1);
      ret_val_symbol.lvalue = true;
      ret_val_symbol.static_lifetime = false;
      ret_val_symbol.location = location;
      ret_val_symbol.mode = original_func.mode;

      log_debug(
        "contracts",
        "generate_checking_wrapper: creating return_value symbol with type "
        "id={}, is_symbol={}",
        ret_val_symbol.get_type().id().as_string(),
        ret_val_symbol.get_type().id() == "symbol");

      // Add symbol to context
      symbolt *added_symbol = context.move_symbol_to_context(ret_val_symbol);
      ret_val = symbol2tc(ret_type, added_symbol->id);

      log_debug(
        "contracts",
        "generate_checking_wrapper: created ret_val symbol, type_id={}, "
        "is_symbol_type={}",
        ret_val->type ? get_type_id(*ret_val->type) : "nil",
        ret_val->type && is_symbol_type(ret_val->type));

      goto_programt::targett decl_inst = wrapper.add_instruction(DECL);
      decl_inst->code = code_decl2tc(ret_type, added_symbol->id);
      decl_inst->location = location;
      decl_inst->location.comment("contract return value");

      log_debug(
        "contracts",
        "generate_checking_wrapper: created DECL instruction, type_id={}, "
        "is_symbol_type={}",
        ret_type ? get_type_id(*ret_type) : "nil",
        ret_type && is_symbol_type(ret_type));

      // Note: We don't initialize return_value here for struct/union types.
      // The function call will assign the complete struct/union value to return_value,
      // which will completely overwrite any member-level initialization.
      // Initializing members individually would be redundant and can cause issues
      // in symbolic execution when the function call overwrites the entire struct.
      //
      // This aligns with the behavior in mark_decl_as_non_det.cpp which skips
      // initialization for return_value$ prefixed variables.
      log_debug(
        "contracts",
        "generate_checking_wrapper: skipping return_value initialization for "
        "struct/union type (will be assigned by function call)");
    }
  }

  // 3. Call original function
  if (original_func.get_type().is_code())
  {
    const code_typet &code_type = to_code_type(original_func.get_type());
    // Convert function type to irep2
    type2tc func_type = migrate_symbol_type(original_func);

    // Build parameter list
    std::vector<expr2tc> arguments;
    const code_typet::argumentst &params = code_type.arguments();
    for (const auto &param : params)
    {
      // Create symbol reference for each parameter
      type2tc param_type = migrate_type(param.type());
      expr2tc param_symbol = symbol2tc(param_type, param.get_identifier());
      arguments.push_back(param_symbol);
    }

    // Create function call
    expr2tc func_symbol = symbol2tc(func_type, original_func_id);
    expr2tc call_expr = code_function_call2tc(ret_val, func_symbol, arguments);

    goto_programt::targett call_inst = wrapper.add_instruction(FUNCTION_CALL);
    call_inst->code = call_expr;
    call_inst->location = location;
    call_inst->location.comment("contract call original function");
  }

  // 3c. Assert assigns compliance (after function call, before ensures)
  if (check_assigns_compliance && declares_frame)
  {
    log_debug(
      "contracts",
      "generate_checking_wrapper: checking assigns compliance for {}",
      id2string(original_func.name));
    // 3c-i. Assert global assigns compliance
    frame_enforcer.enforce_frame_rule(
      assigns_targets, wrapper, location, frame_modet::ASSERT);
    // 3c-ii. Assert ptr->field assigns compliance
    if (!ptr_field_snaps.empty())
      emit_ptr_field_assertions(ptr_field_snaps, wrapper, location);
    // 3c-iii. Phase 2C: assert *p assigns compliance for uncovered pointer params
    if (!ptr_deref_snaps.empty())
      emit_ptr_deref_assertions(ptr_deref_snaps, wrapper, location);
    // 3c-iv. Phase 2B: assert array element assigns compliance
    if (!arr_elem_snaps.empty())
      emit_arr_elem_assertions(arr_elem_snaps, wrapper, location);
  }

  // 4. Assert ensures clause (replace __ESBMC_return_value and __ESBMC_old)
  // Process ensures clause: replace return_value, old(), and is_fresh
  expr2tc ensures_guard = ensures_clause;
  if (!is_nil_expr(ensures_clause))
  {
    log_debug(
      "contracts",
      "generate_checking_wrapper: processing ensures clause, ret_val "
      "type_id={}, is_symbol_type={}",
      ret_val && ret_val->type ? get_type_id(*ret_val->type) : "nil",
      ret_val && ret_val->type && is_symbol_type(ret_val->type));

    // Replace __ESBMC_old() expressions
    if (!old_snapshots.empty())
      ensures_guard = replace_old_in_expr(ensures_guard, old_snapshots);

    // Replace is_fresh temp vars with verification: valid_object(ptr) && is_dynamic[ptr]
    if (!is_fresh_mappings.empty())
      ensures_guard = replace_is_fresh_temps(
        ensures_guard, is_fresh_mappings, /*require_dynamic=*/true);
  }

  // Extract struct member accesses to temporary variables before ASSERT
  // This avoids symbolic execution issues with accessing members from 'with' expressions
  if (!is_nil_expr(ensures_guard) && !is_nil_expr(ret_val))
  {
    log_debug(
      "contracts",
      "Before extract_struct_members_to_temps: ret_val type_id={}, "
      "is_struct={}, is_union={}",
      ret_val->type ? get_type_id(*ret_val->type) : "nil",
      ret_val->type && is_struct_type(ret_val->type),
      ret_val->type && is_union_type(ret_val->type));

    if (is_struct_type(ret_val->type) || is_union_type(ret_val->type))
    {
      ensures_guard = extract_struct_members_to_temps(
        ensures_guard, ret_val, wrapper, location);
    }
  }

  // Normalize ensures guard: replace return_value, fix types, normalize floating-point
  // This unified helper applies all return_value-related transformations
  ensures_guard =
    normalize_ensures_guard_for_return_value(ensures_guard, ret_val);

  // Add ASSERT instruction for ensures clause with property
  if (should_add_clause_instruction(ensures_guard, ASSERT))
  {
    goto_programt::targett t = wrapper.add_instruction(ASSERT);
    t->guard = ensures_guard;
    t->location = location;
    t->location.comment("contract ensures");
    t->location.property("contract ensures");
  }

  // 4b. Free every heap allocation the wrapper performed for the parameters
  //     (is_fresh requires-side mallocs and add_pointer_validity_assumptions
  //     mallocs). Must happen AFTER the ensures assertion — which may
  //     dereference these buffers — and BEFORE the RETURN, so that
  //     --memory-leak-check no longer attributes a CWE-401 forgotten-memory
  //     leak to the user's function (issue #4908).
  for (const expr2tc &ptr : wrapper_heap_ptrs)
  {
    goto_programt::targett free_inst = wrapper.add_instruction(OTHER);
    free_inst->code = code_free2tc(ptr);
    free_inst->location = location;
    free_inst->location.comment("contract: free wrapper-allocated backing");
  }

  // 5. Return the value (if function has return type)
  if (!is_nil_expr(ret_val))
  {
    goto_programt::targett ret_inst = wrapper.add_instruction(RETURN);
    ret_inst->code = code_return2tc(ret_val);
    ret_inst->location = location;
    ret_inst->location.comment("contract return");
  }

  goto_programt::targett end_func = wrapper.add_instruction(END_FUNCTION);
  end_func->location = location;
  return wrapper;
}

expr2tc code_contractst::replace_return_value_in_expr(
  const expr2tc &expr,
  const expr2tc &ret_val) const
{
  if (is_nil_expr(expr))
    return expr;

  // Handle address_of(index(symbol(__ESBMC_return_value))) pattern
  // This is how __ESBMC_return_value appears in GOTO programs when declared as char[]
  if (is_address_of2t(expr))
  {
    const address_of2t &addr = to_address_of2t(expr);
    expr2tc addr_source = addr.ptr_obj;

    if (is_index2t(addr_source))
    {
      const index2t &index = to_index2t(addr_source);
      expr2tc index_source = index.source_value;

      if (is_symbol2t(index_source))
      {
        const symbol2t &sym = to_symbol2t(index_source);
        std::string sym_name = id2string(sym.get_symbol_name());

        if (
          sym_name.find("__ESBMC_return_value") != std::string::npos ||
          sym_name == "return_value")
        {
          // Replace &__ESBMC_return_value[0] with ret_val
          // The original expr is address_of, so its type is pointer
          // But ret_val is the actual return value (not a pointer)
          // We should return ret_val directly, as it has the correct type
          return ret_val;
        }
      }
    }
  }

  // If this is a symbol with name __ESBMC_return_value, replace it
  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);
    std::string sym_name = id2string(sym.get_symbol_name());

    // Check if symbol name contains __ESBMC_return_value (may have prefix)
    // Also check for "return_value" which is how it appears in GOTO programs
    if (
      sym_name.find("__ESBMC_return_value") != std::string::npos ||
      sym_name == "return_value")
    {
      return ret_val;
    }
  }

  // Handle type casting pattern: ((Type*)(&__ESBMC_return_value))->member
  // Pattern: member(dereference(typecast(address_of(index(symbol(__ESBMC_return_value))))))
  // or: member(dereference(typecast(address_of(symbol(__ESBMC_return_value)))))
  if (is_member2t(expr))
  {
    const member2t &member = to_member2t(expr);
    expr2tc source = member.source_value;

    // Check if source is a dereference (for -> operator)
    expr2tc deref_source = source;
    if (is_dereference2t(source))
    {
      const dereference2t &deref = to_dereference2t(source);
      deref_source = deref.value;
    }

    // Check if deref_source (or source if no dereference) is a typecast
    if (is_typecast2t(deref_source))
    {
      const typecast2t &cast = to_typecast2t(deref_source);
      expr2tc cast_source = cast.from;

      // Check if cast source is address_of
      if (is_address_of2t(cast_source))
      {
        const address_of2t &addr = to_address_of2t(cast_source);
        expr2tc addr_source = addr.ptr_obj;

        // Check if address_of source is __ESBMC_return_value symbol (direct or via index)
        expr2tc final_symbol = addr_source;
        if (is_index2t(addr_source))
        {
          // Handle case: ((Type*)(&__ESBMC_return_value[0]))->member
          const index2t &index = to_index2t(addr_source);
          final_symbol = index.source_value;
        }

        if (is_symbol2t(final_symbol))
        {
          const symbol2t &sym = to_symbol2t(final_symbol);
          std::string sym_name = id2string(sym.get_symbol_name());

          if (
            sym_name.find("__ESBMC_return_value") != std::string::npos ||
            sym_name == "return_value")
          {
            // Replace the entire pattern: ((Type*)(&__ESBMC_return_value))->member
            // with: ret_val.member (direct member access)
            // ret_val is already the struct value, not a pointer
            // But we need to check if ret_val is actually a struct/union type
            if (is_struct_type(ret_val->type) || is_union_type(ret_val->type))
            {
              return member2tc(member.type, ret_val, member.member);
            }
            else
            {
              // If ret_val is not a struct/union, we can't create member access
              // This shouldn't happen for struct return types, but handle it gracefully
              log_warning(
                "contracts: cannot create member access: ret_val type is not "
                "struct/union (type={})",
                get_type_id(*ret_val->type));
              // Continue with recursive replacement
            }
          }
        }
      }
      // Handle pointer-return pattern: ((T*)__ESBMC_return_value)->member
      // When the function returns T*, ret_val is T* and cast_source may be
      // the return_value symbol directly, or wrapped in intermediate typecasts
      // (e.g. ((T*)((signed int)return_value))->member when __ESBMC_return_value
      // is declared as extern int).
      else
      {
        // Peel any intermediate typecasts to find the underlying symbol.
        expr2tc inner = cast_source;
        while (is_typecast2t(inner))
          inner = to_typecast2t(inner).from;

        if (is_symbol2t(inner))
        {
          const symbol2t &sym = to_symbol2t(inner);
          std::string sym_name = id2string(sym.get_symbol_name());

          if (
            sym_name.find("__ESBMC_return_value") != std::string::npos ||
            sym_name == "return_value")
          {
            // ret_val holds the actual return value; for pointer-return
            // functions it has type T*. Generate: (*ret_val).member
            if (is_pointer_type(ret_val->type))
            {
              const pointer_type2t &ptr_type = to_pointer_type(ret_val->type);
              // The subtype may be a symbol_type2t — resolve it so that
              // member2tc's assertion (source must be struct/union) passes.
              type2tc pointee = ptr_type.subtype;
              if (is_symbol_type(pointee))
                pointee = ns.follow(pointee);
              if (is_struct_type(pointee) || is_union_type(pointee))
              {
                return member2tc(
                  member.type, dereference2tc(pointee, ret_val), member.member);
              }
            }
          }
        }
      }
    }
  }

  // Recursively replace in all operands
  expr2tc new_expr = expr;
  new_expr->Foreach_operand([this, &ret_val](expr2tc &op) {
    op = replace_return_value_in_expr(op, ret_val);
  });

  // After replacing __ESBMC_return_value, check if we can simplify typecasts
  // If a typecast was added to match __ESBMC_return_value's pointer type,
  // but __ESBMC_return_value is now replaced with ret_val (non-pointer),
  // we may be able to remove the typecast
  if (is_typecast2t(new_expr))
  {
    const typecast2t &cast = to_typecast2t(new_expr);

    // If the cast source type matches the cast target type, remove the typecast
    if (cast.from->type == cast.type)
    {
      return cast.from;
    }

    // If cast target is pointer but cast source is not, and ret_val is not a pointer,
    // this typecast was likely added to match __ESBMC_return_value's pointer type
    // Since we've replaced __ESBMC_return_value with ret_val, we can try to remove it
    if (
      is_pointer_type(cast.type) && !is_pointer_type(cast.from->type) &&
      ret_val && !is_pointer_type(ret_val->type))
    {
      return cast.from;
    }
  }

  return new_expr;
}

expr2tc code_contractst::replace_symbol_in_expr(
  const expr2tc &expr,
  const expr2tc &old_symbol,
  const expr2tc &new_expr) const
{
  if (is_nil_expr(expr))
    return expr;

  // If this is the symbol we want to replace, return the new expression
  if (is_symbol2t(expr) && is_symbol2t(old_symbol))
  {
    const symbol2t &sym = to_symbol2t(expr);
    const symbol2t &old_sym = to_symbol2t(old_symbol);

    // Compare symbol names
    if (sym.thename == old_sym.thename)
    {
      return new_expr;
    }
  }

  // Recursively replace in all operands
  expr2tc result = expr->clone();
  result->Foreach_operand([this, &old_symbol, &new_expr](expr2tc &op) {
    op = replace_symbol_in_expr(op, old_symbol, new_expr);
  });

  return result;
}

expr2tc code_contractst::extract_struct_members_to_temps(
  const expr2tc &expr,
  const expr2tc &ret_val,
  goto_programt &wrapper,
  const locationt &location)
{
  if (is_nil_expr(expr) || is_nil_expr(ret_val) || !is_symbol2t(ret_val))
    return expr;

  const symbol2t &ret_sym = to_symbol2t(ret_val);

  // Map from member name to temporary variable
  std::map<irep_idt, expr2tc> member_to_temp;

  // Recursive function to collect and replace member accesses
  std::function<expr2tc(const expr2tc &)> process_expr =
    [&](const expr2tc &e) -> expr2tc {
    if (is_nil_expr(e))
      return e;

    // Check if this is a member access to ret_val
    if (is_member2t(e))
    {
      const member2t &member = to_member2t(e);

      // Check if the source is ret_val
      if (
        is_symbol2t(member.source_value) &&
        ret_sym.thename == to_symbol2t(member.source_value).thename)
      {
        // Check if we already created a temp for this member
        auto it = member_to_temp.find(member.member);
        if (it != member_to_temp.end())
        {
          return it->second;
        }

        // Create temporary variable for this member
        std::string temp_name =
          id2string(ret_sym.thename) + "$member$" + id2string(member.member);
        irep_idt temp_id(temp_name);

        // Create temporary variable symbol
        symbolt temp_symbol;
        temp_symbol.name = temp_id;
        temp_symbol.id = temp_id;
        set_symbol_type(temp_symbol, member.type);
        temp_symbol.lvalue = true;
        temp_symbol.static_lifetime = false;
        temp_symbol.location = location;
        temp_symbol.mode = "C";

        // Add to context
        symbolt *added_symbol = context.move_symbol_to_context(temp_symbol);
        expr2tc temp_var = symbol2tc(member.type, added_symbol->id);

        // Add DECL instruction
        goto_programt::targett decl_inst = wrapper.add_instruction(DECL);
        decl_inst->code = code_decl2tc(member.type, added_symbol->id);
        decl_inst->location = location;
        decl_inst->location.comment("temp for struct member");

        // Add ASSIGN instruction: temp = ret_val.member
        goto_programt::targett assign_inst = wrapper.add_instruction(ASSIGN);
        assign_inst->code = code_assign2tc(temp_var, e->clone());
        assign_inst->location = location;
        assign_inst->location.comment("extract struct member");

        member_to_temp[member.member] = temp_var;

        log_debug(
          "contracts",
          "extract_struct_members_to_temps: created temp {} for member {}",
          temp_name,
          id2string(member.member));

        return temp_var;
      }
    }

    // Recursively process operands
    expr2tc result = e->clone();
    result->Foreach_operand([&](expr2tc &op) { op = process_expr(op); });

    return result;
  };

  expr2tc result = process_expr(expr);

  log_debug(
    "contracts",
    "extract_struct_members_to_temps: extracted {} members",
    member_to_temp.size());

  return result;
}

// ========== __ESBMC_is_fresh support for ensures implementation ==========

std::vector<code_contractst::is_fresh_mapping_t>
code_contractst::extract_is_fresh_mappings_from_body(
  const goto_programt &function_body) const
{
  std::vector<code_contractst::is_fresh_mapping_t> mappings;

  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_function_call() && is_code_function_call2t(it->code))
    {
      const code_function_call2t &call = to_code_function_call2t(it->code);
      if (
        is_symbol2t(call.function) &&
        is_fresh_function(to_symbol2t(call.function).thename.as_string()) &&
        call.operands.size() >= 2 && !is_nil_expr(call.ret) &&
        is_symbol2t(call.ret))
      {
        code_contractst::is_fresh_mapping_t mapping;
        mapping.temp_var_name = to_symbol2t(call.ret).thename;

        expr2tc ptr_arg = call.operands[0];
        if (is_pointer_type(ptr_arg->type))
        {
          type2tc target_ptr_type = to_pointer_type(ptr_arg->type).subtype;
          if (is_empty_type(target_ptr_type))
            target_ptr_type = pointer_type2tc(get_empty_type());
          mapping.ptr_expr = dereference2tc(target_ptr_type, ptr_arg);
          mappings.push_back(mapping);
        }
      }
    }
  }

  return mappings;
}

// The object a pointer names directly -- `&v`, `&v.f`, `&v.a[i]` -- as opposed
// to one reached through another pointer. Nil for any other shape, including
// `&p[i]` and `&p->f`, whose base symbol is itself a pointer and so says
// nothing about what it points at.
static expr2tc named_base_object(const expr2tc &ptr)
{
  if (!is_address_of2t(ptr))
    return expr2tc();

  expr2tc obj = to_address_of2t(ptr).ptr_obj;
  while (is_member2t(obj) || is_index2t(obj))
    obj = is_member2t(obj) ? to_member2t(obj).source_value
                           : to_index2t(obj).source_value;

  if (!is_symbol2t(obj) || is_pointer_type(obj->type))
    return expr2tc();

  return obj;
}

expr2tc code_contractst::replace_is_fresh_temps(
  const expr2tc &expr,
  const std::vector<is_fresh_mapping_t> &mappings,
  bool require_dynamic) const
{
  if (is_nil_expr(expr))
    return expr;

  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);
    for (const auto &mapping : mappings)
    {
      if (sym.thename == mapping.temp_var_name)
      {
        // Lower is_fresh(p) to a concrete predicate on the pointed-to object.
        //
        // ensures side (require_dynamic == true):
        //   valid_object(p) && is_dynamic[POINTER_OBJECT(p)] -- the
        //   postcondition promises a freshly heap-allocated object.
        //
        // requires side at a --replace-call-with-contract call site
        // (require_dynamic == false):
        //   valid_object(p) and the stated extent, but not is_dynamic. The
        //   precondition is *asserted* against the caller's argument here, and
        //   a real caller legitimately passes a live stack object or an
        //   interior sub-object (e.g. &v->vec[k] of a fresh vector) -- valid,
        //   but not heap-dynamic. Requiring is_dynamic would reject every such
        //   caller and make is_fresh unusable under contract replacement; the
        //   frame guarantee is carried by the callee's assigns clause, not by
        //   heap-freshness. (#6380)
        //
        //   plus the extent the contract asked for, which the caller has to
        //   supply. dynamic_size is a byte count, the currency is_fresh states
        //   its extent in, and the offset term is what keeps
        //   is_fresh(&a[i], n) working. It says nothing about an object with
        //   automatic storage -- __ESBMC_alloc_size is only maintained for
        //   heap objects -- so the extent is owed only where it is meaningful,
        //   which keeps the stack caller above working. (#6542)
        expr2tc valid_obj = valid_object2tc(mapping.ptr_expr);
        expr2tc ptr_obj = pointer_object2tc(pointer_type2(), mapping.ptr_expr);

        const symbolt *dyn_sym = ns.lookup("c:@__ESBMC_is_dynamic");
        if (dyn_sym == nullptr)
        {
          log_error("__ESBMC_is_dynamic symbol not found");
          abort();
        }
        type2tc dyn_arr_type = array_type2tc(get_bool_type(), expr2tc(), true);
        expr2tc dyn_arr = symbol2tc(dyn_arr_type, dyn_sym->id);
        expr2tc is_dynamic = index2tc(get_bool_type(), dyn_arr, ptr_obj);

        if (require_dynamic)
          return and2tc(valid_obj, is_dynamic);

        if (is_nil_expr(mapping.size_expr))
          return valid_obj;

        // off + n <= have would wrap for a large n and pass vacuously, so ask
        // the same question without an addition.
        expr2tc off = typecast2tc(
          size_type2(),
          pointer_offset2tc(
            get_int_type(config.ansi_c.address_width), mapping.ptr_expr));
        expr2tc n = typecast2tc(size_type2(), mapping.size_expr);

        // A named object answers both halves from its declaration rather than
        // from __ESBMC_alloc, which is written for the heap alone. VALID_OBJECT
        // of an automatic or static object is a free boolean a solver may pick
        // false, so a caller passing `&v` could not discharge the precondition
        // at all (#6542); goto-symex/dynamic_allocation.cpp guards
        // invalid_pointer the same way, for the same reason. Dropping the
        // conjunct is not "assume valid": an object is valid for as long as its
        // name is in scope, and this expression was written at the call site.
        // Its extent is the size of its type, which also replaces the
        // DYNAMIC_SIZE the old `!is_dynamic` escape left unchecked.
        expr2tc base = named_base_object(mapping.ptr_expr);
        if (!is_nil_expr(base))
        {
          expr2tc have =
            constant_int2tc(size_type2(), type_byte_size(base->type, &ns));
          return and2tc(
            lessthanequal2tc(off, have),
            lessthanequal2tc(n, sub2tc(size_type2(), have, off)));
        }

        expr2tc have =
          typecast2tc(size_type2(), dynamic_size2tc(mapping.ptr_expr));
        expr2tc fits = and2tc(
          lessthanequal2tc(off, have),
          lessthanequal2tc(n, sub2tc(size_type2(), have, off)));

        return and2tc(valid_obj, or2tc(not2tc(is_dynamic), fits));
      }
    }
  }

  expr2tc new_expr = expr;
  new_expr->Foreach_operand([this, &mappings, require_dynamic](expr2tc &op) {
    op = replace_is_fresh_temps(op, mappings, require_dynamic);
  });

  return new_expr;
}

// ========== __ESBMC_old support implementation ==========

bool code_contractst::is_old_call(const expr2tc &expr) const
{
  if (is_nil_expr(expr))
    return false;

  // Check if this is a sideeffect with kind old_snapshot
  if (is_sideeffect2t(expr))
  {
    const sideeffect2t &se = to_sideeffect2t(expr);
    return se.kind == sideeffect2t::allockind::old_snapshot;
  }

  return false;
}

// A fresh lvalue of \p type registered under \p name.
//
// Note: symbolt uses IRep1 (typet) while we work with IRep2 (type2tc). This is
// ESBMC's architecture: the symbol table is IRep1-based for global state, while
// modern code (GOTO programs, contracts) uses IRep2 for local logic. Set the
// symbol's IREP2 type directly via the migrate-layer chokepoint;
// set_symbol_type stores the cache authoritatively and derives the legacy field
// via migrate_type_back exactly once (esbmc/esbmc#4715, B2 S4b).
expr2tc code_contractst::declare_local_symbol(
  const std::string &name,
  const type2tc &type) const
{
  symbolt symbol;
  symbol.name = name;
  symbol.id = name;
  set_symbol_type(symbol, type);
  symbol.lvalue = true;
  symbol.static_lifetime = false;
  symbol.file_local = false;

  return symbol2tc(type, context.move_symbol_to_context(symbol)->id);
}

expr2tc code_contractst::create_snapshot_variable(
  const expr2tc &expr,
  const std::string &func_name,
  size_t index) const
{
  return declare_local_symbol(
    "__ESBMC_old_snapshot_" + func_name + "_" + std::to_string(index),
    expr->type);
}

expr2tc code_contractst::replace_old_in_expr(
  const expr2tc &expr,
  const std::vector<old_snapshot_t> &snapshots) const
{
  if (is_nil_expr(expr))
    return expr;

  // Handle *(T*)old_raw_temp pattern (from __typeof__ macro: __ESBMC_old_raw approach)
  // The macro expands __ESBMC_old(x) to *(T*)__ESBMC_old_raw(&x), so the ensures
  // expression has dereference(typecast(T*, symbol_with___ESBMC_old_raw_in_name)).
  // We replace the entire dereference-cast-symbol subtree with the snapshot variable.
  if (is_dereference2t(expr))
  {
    const dereference2t &deref = to_dereference2t(expr);
    expr2tc ptr_expr = deref.value;

    // Strip typecasts to find the underlying symbol
    while (is_typecast2t(ptr_expr))
      ptr_expr = to_typecast2t(ptr_expr).from;

    if (is_symbol2t(ptr_expr))
    {
      const symbol2t &sym = to_symbol2t(ptr_expr);
      std::string sym_name = id2string(sym.thename);

      if (sym_name.find("___ESBMC_old") != std::string::npos)
      {
        for (const auto &snapshot : snapshots)
        {
          if (is_symbol2t(snapshot.original_expr))
          {
            const symbol2t &snap_sym = to_symbol2t(snapshot.original_expr);
            if (sym.thename == snap_sym.thename)
              return snapshot.snapshot_var;
          }
        }
      }
    }
  }

  // Check if this is a symbol that matches one of the old temp variables
  // (Legacy path for any direct symbol reference to the old temp var)
  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);
    std::string sym_name = id2string(sym.thename);

    // Only process symbols that are related to __ESBMC_old
    // These temp variables have names containing "___ESBMC_old"
    // This prevents accidentally replacing __ESBMC_return_value or other symbols
    if (sym_name.find("___ESBMC_old") != std::string::npos)
    {
      for (const auto &snapshot : snapshots)
      {
        if (is_symbol2t(snapshot.original_expr))
        {
          const symbol2t &snap_sym = to_symbol2t(snapshot.original_expr);
          if (sym.thename == snap_sym.thename)
          {
            return snapshot.snapshot_var;
          }
        }
      }
    }
  }

  // Check if this is an old_snapshot sideeffect (for compatibility)
  if (is_old_call(expr))
  {
    // Get the expression inside old()
    const sideeffect2t &se = to_sideeffect2t(expr);
    expr2tc original_expr = se.operand;

    // Find matching snapshot
    for (const auto &snapshot : snapshots)
    {
      if (snapshot.original_expr == original_expr)
      {
        return snapshot.snapshot_var;
      }
    }

    log_error("Cannot find snapshot for __ESBMC_old expression");
    abort();
  }

  // Recursively replace in all operands
  expr2tc new_expr = expr->clone();
  new_expr->Foreach_operand([this, &snapshots](expr2tc &op) {
    op = replace_old_in_expr(op, snapshots);
  });

  return new_expr;
}

// ========== Old snapshot collection and materialization helpers ==========

std::vector<code_contractst::old_snapshot_t>
code_contractst::collect_old_snapshots_from_body(
  const goto_programt &function_body) const
{
  std::vector<code_contractst::old_snapshot_t> old_snapshots;

  // Track seen expressions to deduplicate
  // Map: original_expr -> {first_temp_var, all_temp_vars}
  struct expr_info
  {
    expr2tc original_expr;
    std::vector<expr2tc> temp_vars;
  };
  std::vector<expr_info> unique_exprs;

  // Scan for assignments from old_snapshot sideeffects
  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_assign() && is_code_assign2t(it->code))
    {
      const code_assign2t &assign = to_code_assign2t(it->code);
      if (is_sideeffect2t(assign.source))
      {
        const sideeffect2t &se = to_sideeffect2t(assign.source);
        if (se.kind == sideeffect2t::allockind::old_snapshot)
        {
          // Found an old_snapshot assignment!
          // The operand is the original expression, the target is the temp variable
          expr2tc original_expr = se.operand;
          expr2tc temp_var = assign.target;

          // Check if we've seen this expression before
          auto it = std::find_if(
            unique_exprs.begin(),
            unique_exprs.end(),
            [&original_expr](const expr_info &info) {
              return info.original_expr == original_expr;
            });

          if (it != unique_exprs.end())
          {
            // Same expression - add this temp var to the list
            it->temp_vars.push_back(temp_var);
          }
          else
          {
            // New expression - create a new entry
            unique_exprs.push_back({original_expr, {temp_var}});
          }
        }
      }
    }
  }

  // Create one snapshot entry per unique expression, BUT create entries for ALL temp vars
  // This ensures that all temp vars get mapped to the same wrapper snapshot later
  for (const auto &info : unique_exprs)
  {
    // Add an entry for EACH temp var that references this expression
    // They all have the same original_expr, so they'll all get mapped to the same wrapper snapshot
    for (const auto &temp_var : info.temp_vars)
    {
      old_snapshots.push_back({info.original_expr, temp_var});
    }

    // Log if there are multiple temp vars for the same expression
    if (info.temp_vars.size() > 1)
    {
      log_debug(
        "contracts",
        "Found {} temp variables for the same __ESBMC_old expression - all "
        "will map to one snapshot",
        info.temp_vars.size());
    }
  }

  return old_snapshots;
}

// ---------------------------------------------------------------------------
// Pointer-struct-field assigns compliance helpers
// ---------------------------------------------------------------------------

std::vector<code_contractst::ptr_field_snapshot_t>
code_contractst::materialize_ptr_field_snapshots(
  const frame_enforcert::classified_assignst &classified,
  const symbolt &original_func,
  goto_programt &wrapper,
  const locationt &location,
  const std::string &func_name)
{
  std::vector<ptr_field_snapshot_t> result;

  if (classified.ptr_field_targets.empty())
    return result;
  if (!original_func.get_type().is_code())
    return result;

  const code_typet &code_type = to_code_type(original_func.get_type());
  const code_typet::argumentst &params = code_type.arguments();

  for (const auto &[ptr_name, assigned_fields] : classified.ptr_field_targets)
  {
    // Find the parameter whose identifier matches ptr_name
    for (const auto &param : params)
    {
      if (param.get_identifier() != ptr_name)
        continue;

      type2tc param_type = migrate_type(param.type());
      if (!is_pointer_type(param_type))
        break;

      // Resolve the pointed-to struct type
      type2tc pointee_type = to_pointer_type(param_type).subtype;
      if (is_symbol_type(pointee_type))
        pointee_type = ns.follow(pointee_type);
      if (!is_struct_type(pointee_type))
        break;

      const struct_type2t &stype = to_struct_type(pointee_type);
      expr2tc ptr_sym = symbol2tc(param_type, ptr_name);
      expr2tc deref_expr = dereference2tc(pointee_type, ptr_sym);

      for (size_t i = 0; i < stype.member_names.size(); ++i)
      {
        const irep_idt &field = stype.member_names[i];
        if (assigned_fields.count(field))
          continue; // This field is explicitly assigned — skip

        const type2tc &ftype = stype.members[i];

        // Create a uniquely-named snapshot symbol
        std::string snap_name = "__ESBMC_frame_snap_ptrf_" + func_name + "_" +
                                id2string(ptr_name) + "_" + id2string(field) +
                                "_" + std::to_string(ptr_field_snap_counter++);

        symbolt snap_sym_obj;
        snap_sym_obj.name = snap_name;
        snap_sym_obj.id = snap_name;
        set_symbol_type(snap_sym_obj, ftype);
        snap_sym_obj.lvalue = true;
        snap_sym_obj.static_lifetime = false;
        snap_sym_obj.file_local = false;
        symbolt *added = context.move_symbol_to_context(snap_sym_obj);
        expr2tc snap_expr = symbol2tc(ftype, added->id);

        // DECL snapshot variable
        goto_programt::targett decl_inst = wrapper.add_instruction(DECL);
        decl_inst->code = code_decl2tc(ftype, added->id);
        decl_inst->location = location;
        decl_inst->location.comment("frame: ptr-field snapshot declaration");

        // ASSIGN snapshot = ptr->field (pre-call value)
        expr2tc field_expr = member2tc(ftype, deref_expr, field);
        goto_programt::targett assign_inst = wrapper.add_instruction(ASSIGN);
        assign_inst->code = code_assign2tc(snap_expr, field_expr);
        assign_inst->location = location;
        assign_inst->location.comment("frame: capture ptr-field pre-state");

        ptr_field_snapshot_t entry;
        entry.ptr_sym = ptr_sym;
        entry.pointee_type = pointee_type;
        entry.field_name = field;
        entry.field_type = ftype;
        entry.snapshot_sym = snap_expr;
        result.push_back(entry);
      }
      break; // matched the parameter
    }
  }
  return result;
}

void code_contractst::emit_ptr_field_assertions(
  const std::vector<ptr_field_snapshot_t> &snapshots,
  goto_programt &wrapper,
  const locationt &location)
{
  for (const auto &entry : snapshots)
  {
    expr2tc deref_expr = dereference2tc(entry.pointee_type, entry.ptr_sym);
    expr2tc field_expr =
      member2tc(entry.field_type, deref_expr, entry.field_name);
    expr2tc guard = equality2tc(field_expr, entry.snapshot_sym);

    goto_programt::targett t = wrapper.add_instruction(ASSERT);
    t->guard = guard;
    t->location = location;
    std::string label = id2string(to_symbol2t(entry.ptr_sym).thename) + "->" +
                        id2string(entry.field_name);
    t->location.comment(
      "assigns compliance: " + label + " not in assigns clause");
    t->location.property("assigns compliance");
  }
}

// ---------------------------------------------------------------------------
// Phase 2C: pointer-parameter dereference assigns compliance
// ---------------------------------------------------------------------------

void code_contractst::materialize_ptr_deref_array_field(
  const irep_idt &param_id,
  const irep_idt &field,
  const type2tc &ftype,
  const type2tc &pointee,
  const expr2tc &ptr_sym,
  const expr2tc &deref_expr,
  goto_programt &wrapper,
  const locationt &location,
  const std::string &func_name,
  std::vector<ptr_deref_snapshot_t> &result)
{
  const array_type2t &atype = to_array_type(ftype);
  type2tc elem_type = atype.subtype;
  if (is_symbol_type(elem_type))
    elem_type = ns.follow(elem_type);
  // Only constant-size arrays of scalar elements (skip VLAs and nested
  // array/struct elements, which would recurse into the same array-rvalue
  // problem).
  if (!is_constant_int2t(atype.array_size) || !is_scalar_type(elem_type))
    return;
  // A zero-length array member -- the GCC trailing-flexible-member idiom --
  // has no element to snapshot, and no index is valid in it. The witness range
  // used to be assumed, so an empty one assumed `false` ahead of the call and
  // discharged every assertion in the wrapper, verifying the contract
  // vacuously (#6513).
  if (to_constant_int2t(atype.array_size).value == 0)
    return;
  type2tc k_type = atype.array_size->type;

  std::string base = func_name + "_" + id2string(param_id) + "_" +
                     id2string(field) + "_" +
                     std::to_string(ptr_deref_snap_counter++);

  // nondet witness index k, constrained to [0, n)
  symbolt k_obj;
  k_obj.name = k_obj.id = "__ESBMC_frame_pderef_k_" + base;
  set_symbol_type(k_obj, k_type);
  k_obj.lvalue = true;
  k_obj.static_lifetime = false;
  k_obj.file_local = false;
  symbolt *k_added = context.move_symbol_to_context(k_obj);
  expr2tc witness_k = symbol2tc(k_type, k_added->id);

  goto_programt::targett k_decl = wrapper.add_instruction(DECL);
  k_decl->code = code_decl2tc(k_type, k_added->id);
  k_decl->location = location;
  k_decl->location.comment("frame: ptr-deref array witness (Phase 2C)");

  goto_programt::targett k_asg = wrapper.add_instruction(ASSIGN);
  k_asg->code = code_assign2tc(witness_k, gen_nondet(k_type));
  k_asg->location = location;

  // Clamp rather than ASSUME, as Phase 2B does: an assumption over the witness
  // index excludes paths whenever the range is empty. Element 0 exists here
  // because the empty case was skipped above, so it is always a valid fallback
  // (#6513).
  goto_programt::targett k_rng = wrapper.add_instruction(ASSIGN);
  k_rng->code = code_assign2tc(
    witness_k,
    if2tc(
      k_type,
      and2tc(
        greaterthanequal2tc(witness_k, gen_zero(k_type)),
        lessthan2tc(witness_k, atype.array_size)),
      witness_k,
      gen_zero(k_type)));
  k_rng->location = location;
  k_rng->location.comment(
    "frame: clamp ptr-deref index to valid array range (Phase 2C)");

  // scalar snapshot of (*p).field[k]
  symbolt s_obj;
  s_obj.name = s_obj.id = "__ESBMC_frame_snap_pderef_" + base;
  set_symbol_type(s_obj, elem_type);
  s_obj.lvalue = true;
  s_obj.static_lifetime = false;
  s_obj.file_local = false;
  symbolt *s_added = context.move_symbol_to_context(s_obj);
  expr2tc snap_expr = symbol2tc(elem_type, s_added->id);

  goto_programt::targett s_decl = wrapper.add_instruction(DECL);
  s_decl->code = code_decl2tc(elem_type, s_added->id);
  s_decl->location = location;
  s_decl->location.comment("frame: ptr-deref array snapshot (Phase 2C)");

  expr2tc field_arr = member2tc(ftype, deref_expr, field);
  goto_programt::targett s_asg = wrapper.add_instruction(ASSIGN);
  s_asg->code =
    code_assign2tc(snap_expr, index2tc(elem_type, field_arr, witness_k));
  s_asg->location = location;
  s_asg->location.comment("frame: capture (ptr->field)[k] (Phase 2C)");

  ptr_deref_snapshot_t entry;
  entry.ptr_sym = ptr_sym;
  entry.pointee_type = pointee;
  entry.field_name = field;
  entry.value_type = elem_type;
  entry.snapshot_sym = snap_expr;
  entry.array_index = witness_k;
  entry.member_type = ftype; // array type, for member access at assert time
  result.push_back(entry);
}

/// The base pointer of an assigns target that covers \p field, or nil if the
/// target has some other shape. Used to decide whether a snapshotted parameter
/// could be another name for memory the clause permits writing.
///
/// Two shapes qualify. `__ESBMC_assigns(*p)` covers the whole pointee and so
/// every field. `p->field`, optionally indexed, covers just that field, which
/// is what keeps the exemption from reaching a sibling. A bare `p` names the
/// pointer rather than the pointee, so it grants nothing here.
///
/// Note that array decay already lowers `p->arr` to `index(member(...), 0)`
/// upstream, so stripping indices here does not lose a distinction that
/// existed: `assigns(p->arr)` and `assigns(p->arr[0])` reach this identically.
/// Should per-index checking of the declared base ever be tightened, this
/// stripping has to be revisited with it.
static expr2tc assigns_target_base(const expr2tc &target, const irep_idt &field)
{
  expr2tc e = target;
  while (is_typecast2t(e))
    e = to_typecast2t(e).from;

  // `__ESBMC_assigns(*p)`: the whole pointee, hence every field. A bare `p`
  // names the pointer, not what it points at, so it is deliberately not
  // matched: exempting every field of `*p` would waive frame checks the clause
  // never granted.
  if (is_dereference2t(e))
  {
    expr2tc whole = to_dereference2t(e).value;
    while (is_typecast2t(whole))
      whole = to_typecast2t(whole).from;
    return is_symbol2t(whole) ? whole : expr2tc();
  }

  while (is_index2t(e))
    e = to_index2t(e).source_value;
  if (!is_member2t(e) || to_member2t(e).member != field)
    return expr2tc();

  expr2tc src = to_member2t(e).source_value;
  if (!is_dereference2t(src))
    return expr2tc();

  expr2tc ptr = to_dereference2t(src).value;
  while (is_typecast2t(ptr))
    ptr = to_typecast2t(ptr).from;
  return is_symbol2t(ptr) ? ptr : expr2tc();
}

std::vector<code_contractst::ptr_deref_snapshot_t>
code_contractst::materialize_ptr_deref_snapshots(
  const frame_enforcert::classified_assignst &classified,
  const std::vector<expr2tc> &assigns_targets,
  const symbolt &original_func,
  goto_programt &wrapper,
  const locationt &location,
  const std::string &func_name,
  const std::map<irep_idt, param_extentt> &param_extents)
{
  std::vector<ptr_deref_snapshot_t> result;

  // Only check when an assigns clause was explicitly declared.
  // If assigns_targets is empty there is no clause → function may modify anything.
  if (assigns_targets.empty())
    return result;
  if (!original_func.get_type().is_code())
    return result;

  const code_typet &code_type = to_code_type(original_func.get_type());
  const code_typet::argumentst &params = code_type.arguments();

  for (const auto &param : params)
  {
    if (!param.type().is_pointer())
      continue;

    irep_idt param_id = param.get_identifier();

    // The snapshot below reads *p. Against an unjustified backing that read is
    // out of bounds, so the wrapper would report a violation in a parameter
    // the contract never mentions and the body never touches. There is nothing
    // to protect either: the body cannot validly dereference it.
    auto extent_it = param_extents.find(param_id);
    if (extent_it != param_extents.end() && !extent_it->second.justified)
      continue;

    type2tc param_type = migrate_type(param.type());
    type2tc pointee = to_pointer_type(param_type).subtype;
    if (is_symbol_type(pointee))
      pointee = ns.follow(pointee);

    // Skip void*, function pointers, pointer-to-pointer (for now)
    if (
      is_empty_type(pointee) || is_code_type(pointee) || is_nil_type(pointee) ||
      is_pointer_type(pointee))
      continue;

    expr2tc ptr_sym = symbol2tc(param_type, param_id);

    // If *p is fully covered by the assigns clause → skip.
    // Also skip when a pointer-arithmetic target involves this param
    // (e.g. arr[idx] = *(arr+idx) yields pointer_target = arr+idx, which
    // contains param_id as a sub-expression).
    for (const auto &t : classified.pointer_targets)
    {
      // Direct: pointer_target IS the param symbol
      if (is_symbol2t(t) && to_symbol2t(t).thename == param_id)
        goto next_param;

      // Pointer arithmetic (arr+idx, arr-off, ...): check both operands
      // arr[idx] compiles to *(arr+idx), so pointer_target = arr + idx.
      // We check whether the base operand is the param.
      auto has_param_base = [&param_id](const expr2tc &e) -> bool {
        const expr2tc *lhs = nullptr;
        if (is_add2t(e))
          lhs = &to_add2t(e).side_1;
        else if (is_sub2t(e))
          lhs = &to_sub2t(e).side_1;
        if (lhs && is_symbol2t(*lhs) && to_symbol2t(*lhs).thename == param_id)
          return true;
        // Also check side_2 in case compiler swapped operands
        const expr2tc *rhs = nullptr;
        if (is_add2t(e))
          rhs = &to_add2t(e).side_2;
        if (rhs && is_symbol2t(*rhs) && to_symbol2t(*rhs).thename == param_id)
          return true;
        return false;
      };
      if (has_param_base(t))
        goto next_param;
    }

    // If specific fields of *p are declared → already handled by ptr_field_snaps
    if (classified.ptr_field_targets.count(param_id))
      continue;

    // Skip if any direct_target contains this param symbol.
    // This handles patterns like p[0].x, p[1].y in assigns which are
    // classified as direct_targets (member through pointer-arithmetic dereference).
    // In that case the assigns clause already covers writes through p at some
    // index, so Phase 2C's whole-object snapshot would produce false positives.
    {
      // Recursively search for param_id symbol in an expression tree.
      std::function<bool(const expr2tc &)> contains_param =
        [&](const expr2tc &e) -> bool {
        if (!e)
          return false;
        if (is_symbol2t(e) && to_symbol2t(e).thename == param_id)
          return true;
        for (size_t i = 0; i < e->get_num_sub_exprs(); ++i)
          if (contains_param(*e->get_sub_expr(i)))
            return true;
        return false;
      };
      if (std::any_of(
            classified.direct_targets.begin(),
            classified.direct_targets.end(),
            contains_param))
        goto next_param;
    }

    // This pointer param is NOT in the assigns clause at all → snapshot *p.
    if (is_structure_type(pointee))
    {
      // Snapshot each scalar field of the struct.
      const struct_type2t &stype = to_struct_type(pointee);
      expr2tc deref_expr = dereference2tc(pointee, ptr_sym);

      for (size_t fi = 0; fi < stype.member_names.size(); ++fi)
      {
        const irep_idt &field = stype.member_names[fi];
        const type2tc &ftype = stype.members[fi];
        // Skip pointer and function-type fields to avoid complications
        if (is_pointer_type(ftype) || is_code_type(ftype))
          continue;

        if (is_array_type(ftype))
        {
          materialize_ptr_deref_array_field(
            param_id,
            field,
            ftype,
            pointee,
            ptr_sym,
            deref_expr,
            wrapper,
            location,
            func_name,
            result);
          continue;
        }

        std::string snap_name = "__ESBMC_frame_snap_pderef_" + func_name + "_" +
                                id2string(param_id) + "_" + id2string(field) +
                                "_" + std::to_string(ptr_deref_snap_counter++);

        symbolt snap_obj;
        snap_obj.name = snap_name;
        snap_obj.id = snap_name;
        set_symbol_type(snap_obj, ftype);
        snap_obj.lvalue = true;
        snap_obj.static_lifetime = false;
        snap_obj.file_local = false;
        symbolt *added = context.move_symbol_to_context(snap_obj);
        expr2tc snap_expr = symbol2tc(ftype, added->id);

        goto_programt::targett decl_t = wrapper.add_instruction(DECL);
        decl_t->code = code_decl2tc(ftype, added->id);
        decl_t->location = location;
        decl_t->location.comment("frame: ptr-deref field snapshot (Phase 2C)");

        expr2tc field_expr = member2tc(ftype, deref_expr, field);
        goto_programt::targett assign_t = wrapper.add_instruction(ASSIGN);
        assign_t->code = code_assign2tc(snap_expr, field_expr);
        assign_t->location = location;
        assign_t->location.comment(
          "frame: capture ptr->field pre-state (Phase 2C)");

        ptr_deref_snapshot_t entry;
        entry.ptr_sym = ptr_sym;
        entry.pointee_type = pointee;
        entry.field_name = field;
        entry.value_type = ftype;
        entry.snapshot_sym = snap_expr;
        result.push_back(entry);
      }
    }
    else
    {
      // Scalar pointee: snapshot *p directly.
      std::string snap_name = "__ESBMC_frame_snap_pderef_" + func_name + "_" +
                              id2string(param_id) + "_" +
                              std::to_string(ptr_deref_snap_counter++);

      symbolt snap_obj;
      snap_obj.name = snap_name;
      snap_obj.id = snap_name;
      set_symbol_type(snap_obj, pointee);
      snap_obj.lvalue = true;
      snap_obj.static_lifetime = false;
      snap_obj.file_local = false;
      symbolt *added = context.move_symbol_to_context(snap_obj);
      expr2tc snap_expr = symbol2tc(pointee, added->id);

      goto_programt::targett decl_t = wrapper.add_instruction(DECL);
      decl_t->code = code_decl2tc(pointee, added->id);
      decl_t->location = location;
      decl_t->location.comment("frame: ptr-deref snapshot (Phase 2C)");

      expr2tc deref_expr = dereference2tc(pointee, ptr_sym);
      goto_programt::targett assign_t = wrapper.add_instruction(ASSIGN);
      assign_t->code = code_assign2tc(snap_expr, deref_expr);
      assign_t->location = location;
      assign_t->location.comment("frame: capture *ptr pre-state (Phase 2C)");

      ptr_deref_snapshot_t entry;
      entry.ptr_sym = ptr_sym;
      entry.pointee_type = pointee;
      // field_name left empty → scalar dereference
      entry.value_type = pointee;
      entry.snapshot_sym = snap_expr;
      result.push_back(entry);
    }

  next_param:;
  }

  // A parameter that aliases an assigns target's base pointer is another name
  // for the same memory, so a write the clause permits necessarily shows up
  // under both names. Exempt exactly that, matched on the field so a sibling
  // field stays protected. Without it the aliasing introduced for #6551 turns
  // sound contracts into spurious frame violations: the mlk_poly_add(r, b)
  // shape called as add(p, p) writes only r->coeffs, which it does declare.
  attach_alias_exemptions(
    result, assigns_targets, original_func, wrapper, location, func_name);

  return result;
}

/// \brief Mark snapshots whose location an assigns target may also name.
///
/// Pointer parameters may alias (see emit_pointer_param_aliasing), so a
/// parameter can be another name for memory the clause permits writing. The
/// frame assertion would then report a violation that is not one. Each
/// exemption is matched on the field, so a sibling field stays protected.
void code_contractst::attach_alias_exemptions(
  std::vector<ptr_deref_snapshot_t> &result,
  const std::vector<expr2tc> &assigns_targets,
  const symbolt &original_func,
  goto_programt &wrapper,
  const locationt &location,
  const std::string &func_name)
{
  // One snapshot per distinct base, not per (snapshot, target) pair: the same
  // base recurs across targets and across snapshots, and each extra copy is a
  // symbol plus a DECL and an ASSIGN in every wrapper.
  //
  // The base has to be read in the pre-state, because the exemption is
  // asserted after the call while the value it guards was snapshotted before
  // it. A callee free to assign a global base could otherwise point it at the
  // checked object on the way out and launder a write that was outside the
  // frame when it happened.
  std::map<irep_idt, expr2tc> base_snapshots;
  auto snapshot_of = [&](const expr2tc &base) {
    const irep_idt key = to_symbol2t(base).thename;
    auto it = base_snapshots.find(key);
    if (it != base_snapshots.end())
      return it->second;

    std::string base_name = "__ESBMC_frame_aliasbase_" + func_name + "_" +
                            get_pretty_name(id2string(key));
    symbolt base_sym;
    base_sym.name = base_name;
    base_sym.id = base_name;
    set_symbol_type(base_sym, base->type);
    base_sym.lvalue = true;
    base_sym.static_lifetime = false;
    base_sym.location = location;
    base_sym.mode = original_func.mode;
    const irep_idt base_id = context.move_symbol_to_context(base_sym)->id;
    expr2tc snapshot = symbol2tc(base->type, base_id);

    auto base_decl = wrapper.add_instruction(DECL);
    base_decl->code = code_decl2tc(base->type, base_id);
    base_decl->location = location;
    base_decl->location.comment("frame: assigns-target base, pre-state");

    auto base_assign = wrapper.add_instruction(ASSIGN);
    base_assign->code = code_assign2tc(snapshot, base);
    base_assign->location = location;
    base_assign->location.comment("frame: assigns-target base, pre-state");

    base_snapshots.emplace(key, snapshot);
    return snapshot;
  };

  for (auto &snap : result)
  {
    expr2tc exemption;
    for (const expr2tc &target : assigns_targets)
    {
      expr2tc base = assigns_target_base(target, snap.field_name);
      if (is_nil_expr(base) || base == snap.ptr_sym)
        continue;

      expr2tc same = same_object2tc(snap.ptr_sym, snapshot_of(base));
      exemption = is_nil_expr(exemption) ? same : or2tc(exemption, same);
    }
    snap.alias_exemption = exemption;
  }
}

void code_contractst::emit_ptr_deref_assertions(
  const std::vector<ptr_deref_snapshot_t> &snapshots,
  goto_programt &wrapper,
  const locationt &location)
{
  for (const auto &snap : snapshots)
  {
    expr2tc current_val;
    std::string var_label;

    if (snap.field_name.empty())
    {
      // Scalar: assert *p == snapshot
      current_val = dereference2tc(snap.value_type, snap.ptr_sym);
      var_label = "*" + id2string(to_symbol2t(snap.ptr_sym).thename);
    }
    else
    {
      // Struct field: assert p->field == snapshot
      expr2tc deref_expr = dereference2tc(snap.pointee_type, snap.ptr_sym);
      if (snap.array_index)
      {
        // Array field: compare the scalar element at the nondet witness index.
        // The member's array type was captured at snapshot time, so we need not
        // rediscover it from the (possibly typedef'd) struct.
        expr2tc field_arr =
          member2tc(snap.member_type, deref_expr, snap.field_name);
        current_val = index2tc(snap.value_type, field_arr, snap.array_index);
        var_label = id2string(to_symbol2t(snap.ptr_sym).thename) + "->" +
                    id2string(snap.field_name) +
                    "[k] (k: nondet witness index)";
      }
      else
      {
        current_val = member2tc(snap.value_type, deref_expr, snap.field_name);
        var_label = id2string(to_symbol2t(snap.ptr_sym).thename) + "->" +
                    id2string(snap.field_name);
      }
    }

    expr2tc guard = equality2tc(current_val, snap.snapshot_sym);
    if (!is_nil_expr(snap.alias_exemption))
      guard = or2tc(guard, snap.alias_exemption);

    goto_programt::targett t = wrapper.add_instruction(ASSERT);
    t->guard = guard;
    t->location = location;
    t->location.comment(
      "assigns compliance: " + var_label + " not in assigns clause");
    t->location.property("assigns compliance");
  }
}

// ========== Phase 2B: array element assigns compliance ==========

std::vector<code_contractst::arr_elem_snapshot_t>
code_contractst::materialize_arr_elem_snapshots(
  const frame_enforcert::classified_assignst &classified,
  const std::vector<expr2tc> &assigns_targets,
  goto_programt &wrapper,
  const locationt &location,
  const std::string &func_name,
  const std::map<irep_idt, param_extentt> &param_extents)
{
  std::vector<arr_elem_snapshot_t> result;

  if (assigns_targets.empty())
    return result;

  for (const auto &t : classified.pointer_targets)
  {
    // Only process pointer-arithmetic targets: add2t(arr_sym, idx_expr)
    if (!is_add2t(t))
      continue;

    const add2t &add = to_add2t(t);

    // Identify which side is the array pointer and which is the index
    expr2tc arr_ptr, idx_expr;
    if (is_pointer_type(add.side_1) && is_symbol2t(add.side_1))
    {
      arr_ptr = add.side_1;
      idx_expr = add.side_2;
    }
    else if (is_pointer_type(add.side_2) && is_symbol2t(add.side_2))
    {
      arr_ptr = add.side_2;
      idx_expr = add.side_1;
    }
    else
    {
      // Complex pointer expression — skip
      continue;
    }

    // Resolve element type
    const pointer_type2t &ptr_type = to_pointer_type(arr_ptr->type);
    type2tc elem_type = ptr_type.subtype;
    if (is_symbol_type(elem_type))
      elem_type = ns.follow(elem_type);

    // Skip void, function, or pointer element types
    if (
      is_empty_type(elem_type) || is_code_type(elem_type) ||
      is_nil_type(elem_type) || is_pointer_type(elem_type))
      continue;

    type2tc j_type = idx_expr->type;
    std::string cnt_str = std::to_string(arr_elem_snap_counter);
    const irep_idt &arr_id = to_symbol2t(arr_ptr).thename;
    const std::string &arr_name = id2string(arr_id);

    // Create nondet witness index j
    std::string j_sym_name =
      "__ESBMC_frame_arr_j_" + func_name + "_" + arr_name + "_" + cnt_str;

    symbolt j_obj;
    j_obj.name = j_sym_name;
    j_obj.id = j_sym_name;
    set_symbol_type(j_obj, j_type);
    j_obj.lvalue = true;
    j_obj.static_lifetime = false;
    j_obj.file_local = false;
    symbolt *j_added = context.move_symbol_to_context(j_obj);
    expr2tc witness_j = symbol2tc(j_type, j_added->id);

    goto_programt::targett j_decl = wrapper.add_instruction(DECL);
    j_decl->code = code_decl2tc(j_type, j_added->id);
    j_decl->location = location;
    j_decl->location.comment("frame: array-elem witness index (Phase 2B)");

    goto_programt::targett j_assign = wrapper.add_instruction(ASSIGN);
    j_assign->code = code_assign2tc(witness_j, gen_nondet(j_type));
    j_assign->location = location;
    j_assign->location.comment("frame: nondet witness index (Phase 2B)");

    // Bound j to the allocated range so that arr[j] is a valid access.
    // Over-bounding j makes arr[j] read past the real allocation and trips a
    // spurious "array bounds violated" (#5314), so prefer the recorded extent
    // and fall back to the constant only when there is none (e.g. globals).
    auto extent_it = param_extents.find(arr_id);
    BigInt elem_sz = type_byte_size(elem_type, &ns);
    expr2tc j_hi = constant_int2tc(j_type, BigInt(WITNESS_IDX_FALLBACK_ELEMS));
    if (extent_it != param_extents.end() && elem_sz > 0)
    {
      // Divide in the extent's own unsigned type before casting, so the
      // quotient is not computed on a truncated value. An extent above
      // LONG_MAX still casts negative, which degrades to the clamp fallback.
      const expr2tc &bytes = extent_it->second.bytes;
      j_hi = typecast2tc(
        j_type,
        div2tc(bytes->type, bytes, constant_int2tc(bytes->type, elem_sz)));
      simplify(j_hi);
    }

    // Clamp rather than ASSUME. The range can be empty -- a zero or
    // sub-element extent, or a symbolic one the solver may pick 0 for -- and a
    // straight-line ASSUME of an empty range discharges every assertion after
    // it, verifying the whole function vacuously. Assuming a non-empty range
    // instead forces the extent to be at least one element, which is #6212 in
    // another guise. Clamping to the declared index does neither.
    // Phase 2C takes the same approach, skipping a zero-length member outright
    // and clamping to element 0 otherwise (#6513).
    goto_programt::targett j_clamp = wrapper.add_instruction(ASSIGN);
    j_clamp->code = code_assign2tc(
      witness_j,
      if2tc(
        j_type,
        and2tc(
          greaterthanequal2tc(witness_j, gen_zero(j_type)),
          lessthan2tc(witness_j, j_hi)),
        witness_j,
        idx_expr));
    j_clamp->location = location;
    j_clamp->location.comment(
      "frame: clamp witness index to valid array range (Phase 2B)");

    // Create snapshot arr[j] = *(arr + j)
    std::string snap_sym_name =
      "__ESBMC_frame_arr_snap_" + func_name + "_" + arr_name + "_" + cnt_str;

    symbolt snap_obj;
    snap_obj.name = snap_sym_name;
    snap_obj.id = snap_sym_name;
    set_symbol_type(snap_obj, elem_type);
    snap_obj.lvalue = true;
    snap_obj.static_lifetime = false;
    snap_obj.file_local = false;
    symbolt *snap_added = context.move_symbol_to_context(snap_obj);
    expr2tc snapshot_sym = symbol2tc(elem_type, snap_added->id);

    // arr + j (pointer arithmetic, same result type as arr + idx)
    type2tc arr_add_type = add.type;
    expr2tc arr_plus_j = add2tc(arr_add_type, arr_ptr, witness_j);
    expr2tc arr_at_j = dereference2tc(elem_type, arr_plus_j);

    goto_programt::targett snap_decl = wrapper.add_instruction(DECL);
    snap_decl->code = code_decl2tc(elem_type, snap_added->id);
    snap_decl->location = location;
    snap_decl->location.comment("frame: array-elem snapshot (Phase 2B)");

    goto_programt::targett snap_assign = wrapper.add_instruction(ASSIGN);
    snap_assign->code = code_assign2tc(snapshot_sym, arr_at_j);
    snap_assign->location = location;
    snap_assign->location.comment("frame: capture arr[j] pre-state (Phase 2B)");

    arr_elem_snapshot_t entry;
    entry.arr_ptr = arr_ptr;
    entry.arr_add_type = arr_add_type;
    entry.elem_type = elem_type;
    entry.declared_idx = idx_expr;
    entry.witness_idx = witness_j;
    entry.snapshot_sym = snapshot_sym;
    result.push_back(entry);

    arr_elem_snap_counter++;
  }

  return result;
}

void code_contractst::emit_arr_elem_assertions(
  const std::vector<arr_elem_snapshot_t> &snapshots,
  goto_programt &wrapper,
  const locationt &location)
{
  for (const auto &snap : snapshots)
  {
    // Re-read arr[j] after the call (same j, new SSA version of arr)
    expr2tc arr_plus_j =
      add2tc(snap.arr_add_type, snap.arr_ptr, snap.witness_idx);
    expr2tc arr_at_j_after = dereference2tc(snap.elem_type, arr_plus_j);

    // Guard: (j == declared_idx) || (arr[j] == snap)
    expr2tc eq_idx = equality2tc(snap.witness_idx, snap.declared_idx);
    expr2tc eq_val = equality2tc(arr_at_j_after, snap.snapshot_sym);
    expr2tc guard = or2tc(eq_idx, eq_val);

    goto_programt::targett t = wrapper.add_instruction(ASSERT);
    t->guard = guard;
    t->location = location;
    std::string arr_name = id2string(to_symbol2t(snap.arr_ptr).thename);
    t->location.comment(
      "assigns compliance: " + arr_name +
      "[j] not in assigns clause (Phase 2B)");
    t->location.property("assigns compliance");
  }
}

void code_contractst::materialize_old_snapshots_at_wrapper(
  std::vector<code_contractst::old_snapshot_t> &old_snapshots,
  goto_programt &wrapper,
  const std::string &func_name,
  const locationt &location) const
{
  // Generate snapshot assignments in the wrapper BEFORE calling the original function
  // We'll update old_snapshots to contain new wrapper snapshot variables

  // Map to track: original_expr -> wrapper_snapshot_var
  // This ensures we only create ONE wrapper snapshot per unique expression.
  // Uses expr2tc semantic comparison (operator< / operator==) instead of
  // raw pointer addresses, so structurally equal expressions share a snapshot.
  std::map<expr2tc, expr2tc> expr_to_wrapper_snapshot;

  size_t unique_snapshot_count = 0;

  for (size_t i = 0; i < old_snapshots.size(); ++i)
  {
    expr2tc original_expr = old_snapshots[i].original_expr;
    expr2tc old_temp_var =
      old_snapshots[i].snapshot_var; // The temp var from function body

    expr2tc new_snapshot_var;

    // Check if we've already created a wrapper snapshot for this expression
    auto it = expr_to_wrapper_snapshot.find(original_expr);
    if (it != expr_to_wrapper_snapshot.end())
    {
      // Reuse existing wrapper snapshot
      new_snapshot_var = it->second;
      log_debug(
        "contracts",
        "Reusing wrapper snapshot for duplicate __ESBMC_old expression");
    }
    else
    {
      // Create a NEW snapshot variable for the wrapper
      new_snapshot_var = create_snapshot_variable(
        original_expr, func_name + "_wrapper", unique_snapshot_count++);

      // Generate snapshot declaration
      goto_programt::targett decl_inst = wrapper.add_instruction(DECL);
      decl_inst->code = code_decl2tc(
        original_expr->type, to_symbol2t(new_snapshot_var).thename);
      decl_inst->location = location;
      decl_inst->location.comment("__ESBMC_old snapshot declaration");

      // Generate snapshot assignment: new_snapshot_var = original_expr
      goto_programt::targett assign_inst = wrapper.add_instruction(ASSIGN);
      assign_inst->code = code_assign2tc(new_snapshot_var, original_expr);
      assign_inst->location = location;
      assign_inst->location.comment("__ESBMC_old snapshot assignment");

      // Remember this mapping
      expr_to_wrapper_snapshot[original_expr] = new_snapshot_var;
    }

    // Store both old and new variables in the snapshot structure
    // We'll keep the old temp var as original_expr for matching,
    // and new snapshot var as snapshot_var for replacement
    old_snapshots[i].original_expr = old_temp_var;    // What to find
    old_snapshots[i].snapshot_var = new_snapshot_var; // What to replace with
  }
}

std::vector<code_contractst::old_snapshot_t>
code_contractst::materialize_old_snapshots_at_callsite(
  const std::vector<code_contractst::old_snapshot_t> &old_snapshots,
  const symbolt &function_symbol,
  const std::vector<expr2tc> &actual_args,
  goto_programt &replacement,
  const locationt &call_location) const
{
  std::vector<old_snapshot_t> callsite_snapshots;

  // For each old() in the original body, create a call-site snapshot:
  //   - Evaluate the original expression with actual arguments
  //   - Store it in a fresh snapshot variable before havoc
  //   - Remember mapping from the original temp variable to the snapshot
  for (size_t i = 0; i < old_snapshots.size(); ++i)
  {
    expr2tc original_expr = old_snapshots[i].original_expr;
    expr2tc temp_var = old_snapshots[i].snapshot_var; // temp var from body

    // Apply the same parameter substitution used for requires/ensures
    if (function_symbol.get_type().is_code())
    {
      const code_typet &code_type = to_code_type(function_symbol.get_type());
      const code_typet::argumentst &params = code_type.arguments();

      for (size_t j = 0; j < params.size() && j < actual_args.size(); ++j)
      {
        irep_idt param_id = params[j].get_identifier();
        expr2tc param_expr =
          symbol2tc(migrate_type(params[j].type()), param_id);
        original_expr =
          replace_symbol_in_expr(original_expr, param_expr, actual_args[j]);
      }
    }

    // Create a NEW snapshot variable for the call site
    expr2tc snapshot_var = create_snapshot_variable(
      original_expr, id2string(function_symbol.name) + "_call", i);

    // Generate snapshot declaration at call site
    goto_programt::targett decl_inst = replacement.add_instruction(DECL);
    decl_inst->code =
      code_decl2tc(original_expr->type, to_symbol2t(snapshot_var).thename);
    decl_inst->location = call_location;
    decl_inst->location.comment("__ESBMC_old call-site snapshot declaration");

    // Generate snapshot assignment: snapshot_var = original_expr
    goto_programt::targett assign_inst = replacement.add_instruction(ASSIGN);
    assign_inst->code = code_assign2tc(snapshot_var, original_expr);
    assign_inst->location = call_location;
    assign_inst->location.comment("__ESBMC_old call-site snapshot assignment");

    // Store mapping: temp var from original body -> call-site snapshot var.
    code_contractst::old_snapshot_t snap_entry;
    snap_entry.original_expr = temp_var;    // what to find in ensures
    snap_entry.snapshot_var = snapshot_var; // what to replace with
    callsite_snapshots.push_back(snap_entry);
  }

  if (!callsite_snapshots.empty())
  {
    log_debug(
      "contracts",
      "materialize_old_snapshots_at_callsite: created {} __ESBMC_old call-site "
      "snapshot(s) for function {}",
      callsite_snapshots.size(),
      id2string(function_symbol.name));
  }

  return callsite_snapshots;
}

// ========== Type fixing for return value comparisons ==========

bool code_contractst::is_return_value_symbol(const symbol2t &sym) const
{
  std::string name = id2string(sym.thename);

  // Match various return value patterns:
  // - "return_value"
  // - "__ESBMC_return_value"
  // - "return_value$..." (with suffix)
  // - "tag-return_value$..." (with tag prefix)
  if (name == "return_value" || name == "__ESBMC_return_value")
    return true;

  if (name.find("return_value") != std::string::npos)
    return true;

  return false;
}

expr2tc code_contractst::remove_incorrect_casts(
  const expr2tc &expr,
  const expr2tc &ret_val) const
{
  if (is_nil_expr(expr) || is_nil_expr(ret_val))
    return expr;

  // Strip a chain of typecasts wrapping a return_value symbol.
  //
  // When __ESBMC_return_value is undeclared in C source, Clang assigns it the
  // implicit-int type.  An expression like (size_t)__ESBMC_return_value then
  // compiles to (size_t)(int)rv — two nested casts.  After the symbol is
  // replaced by the actual ret_val (e.g. void*), we need to remove ALL
  // intermediate casts whose type disagrees with ret_val's type, not just the
  // outermost one.
  if (is_typecast2t(expr))
  {
    // Peel off typecasts until we reach a non-cast expression.
    expr2tc inner = expr;
    while (is_typecast2t(inner))
      inner = to_typecast2t(inner).from;

    // If the innermost expression is a return_value symbol, discard all casts.
    if (is_symbol2t(inner) && is_return_value_symbol(to_symbol2t(inner)))
    {
      const typecast2t &outermost = to_typecast2t(expr);
      if (!base_type_eq(outermost.type, ret_val->type, ns))
      {
        log_debug(
          "contracts",
          "Removing cast chain down to return_value symbol "
          "(outer cast type={}, ret_val type={})",
          get_type_id(*outermost.type),
          get_type_id(*ret_val->type));
        return inner;
      }
    }
  }

  return expr;
}

/// Helper: extract mutable pointers to side_1 and side_2 of any comparison
/// expression.  Returns false if the expression is not a comparison.
static bool
get_comparison_sides(expr2tc &expr, expr2tc *&side1, expr2tc *&side2)
{
  side1 = side2 = nullptr;
  if (is_lessthan2t(expr))
  {
    auto &r = to_lessthan2t(expr);
    side1 = &r.side_1;
    side2 = &r.side_2;
  }
  else if (is_lessthanequal2t(expr))
  {
    auto &r = to_lessthanequal2t(expr);
    side1 = &r.side_1;
    side2 = &r.side_2;
  }
  else if (is_greaterthan2t(expr))
  {
    auto &r = to_greaterthan2t(expr);
    side1 = &r.side_1;
    side2 = &r.side_2;
  }
  else if (is_greaterthanequal2t(expr))
  {
    auto &r = to_greaterthanequal2t(expr);
    side1 = &r.side_1;
    side2 = &r.side_2;
  }
  else if (is_equality2t(expr))
  {
    auto &r = to_equality2t(expr);
    side1 = &r.side_1;
    side2 = &r.side_2;
  }
  else if (is_notequal2t(expr))
  {
    auto &r = to_notequal2t(expr);
    side1 = &r.side_1;
    side2 = &r.side_2;
  }
  return side1 != nullptr;
}

expr2tc code_contractst::fix_comparison_types(
  const expr2tc &expr,
  const expr2tc &ret_val) const
{
  if (is_nil_expr(expr) || is_nil_expr(ret_val))
    return expr;

  // NON-RECURSIVE APPROACH: Use explicit stack-based traversal to avoid infinite loops
  // We only need to fix comparison expressions, so we traverse the tree explicitly
  // and only process comparison nodes and their direct children

  // Step 1: Handle top-level comparison expressions
  if (is_comp_expr(expr))
  {
    expr2tc new_expr = expr->clone();

    // Get the two sides of the comparison
    expr2tc *side1 = nullptr;
    expr2tc *side2 = nullptr;
    get_comparison_sides(new_expr, side1, side2);

    if (side1 && side2)
    {
      // Step 1a: Remove incorrect casts on direct return_value symbols
      *side1 = remove_incorrect_casts(*side1, ret_val);
      *side2 = remove_incorrect_casts(*side2, ret_val);

      // Step 1b: Handle nested typecasts wrapping add/sub expressions
      // Example: (double)(old_snapshot + (signed int)return_value)
      // Only process one level: typecast -> add/sub -> operands
      for (expr2tc *side_ptr : {side1, side2})
      {
        if (is_typecast2t(*side_ptr))
        {
          const typecast2t &cast = to_typecast2t(*side_ptr);
          expr2tc inner = cast.from;

          // If inner is add/sub, fix its operands (one level only)
          if (is_add2t(inner))
          {
            const add2t &add = to_add2t(inner);
            expr2tc fixed_op1 = remove_incorrect_casts(add.side_1, ret_val);
            expr2tc fixed_op2 = remove_incorrect_casts(add.side_2, ret_val);

            // Only recreate if something changed
            if (fixed_op1 != add.side_1 || fixed_op2 != add.side_2)
            {
              // For floating-point types, use IEEE addition instead of regular addition
              // This matches how floating-point operations are compiled in the actual code
              if (is_fractional_type(cast.type))
              {
                // Use IEEE floating-point addition with default rounding mode
                expr2tc rounding_mode =
                  symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode");
                expr2tc new_add =
                  ieee_add2tc(cast.type, fixed_op1, fixed_op2, rounding_mode);
                *side_ptr =
                  new_add; // No need for outer typecast, ieee_add already has correct type
              }
              else
              {
                // For non-floating-point types, use regular addition
                type2tc add_type = inner->type;
                if (
                  fixed_op1->type == fixed_op2->type &&
                  fixed_op1->type == cast.type)
                {
                  add_type = cast.type;
                }
                expr2tc new_add = add2tc(add_type, fixed_op1, fixed_op2);
                *side_ptr = typecast2tc(cast.type, new_add);
              }
            }
          }
          else if (is_sub2t(inner))
          {
            const sub2t &sub = to_sub2t(inner);
            expr2tc fixed_op1 = remove_incorrect_casts(sub.side_1, ret_val);
            expr2tc fixed_op2 = remove_incorrect_casts(sub.side_2, ret_val);

            if (fixed_op1 != sub.side_1 || fixed_op2 != sub.side_2)
            {
              type2tc sub_type = inner->type;
              expr2tc new_sub = sub2tc(sub_type, fixed_op1, fixed_op2);
              *side_ptr = typecast2tc(cast.type, new_sub);
            }
          }
          else
          {
            // Simple typecast - fix inner expression
            expr2tc fixed = remove_incorrect_casts(inner, ret_val);
            if (fixed != inner)
            {
              *side_ptr = typecast2tc(cast.type, fixed);
            }
          }
        }
      }

      // Step 1c: Check if one side is return_value and fix type mismatches
      bool side1_is_retval =
        is_symbol2t(*side1) && is_return_value_symbol(to_symbol2t(*side1));
      bool side2_is_retval =
        is_symbol2t(*side2) && is_return_value_symbol(to_symbol2t(*side2));

      // Case 1: return_value compared with integer constant, but return_value is pointer
      if (is_pointer_type(ret_val->type))
      {
        if (side1_is_retval && is_constant_int2t(*side2))
        {
          const constant_int2t &c = to_constant_int2t(*side2);
          if (c.value.is_zero())
          {
            *side2 = gen_zero(ret_val->type);
            log_debug(
              "contracts", "Fixed pointer comparison: replaced 0 with NULL");
          }
        }
        else if (side2_is_retval && is_constant_int2t(*side1))
        {
          const constant_int2t &c = to_constant_int2t(*side1);
          if (c.value.is_zero())
          {
            *side1 = gen_zero(ret_val->type);
            log_debug(
              "contracts", "Fixed pointer comparison: replaced 0 with NULL");
          }
        }
      }
      // Case 2: return_value is float/double, constant needs cast
      else if (is_fractional_type(ret_val->type))
      {
        if (side1_is_retval && is_constant_int2t(*side2))
        {
          *side2 = typecast2tc(ret_val->type, *side2);
          log_debug(
            "contracts",
            "Fixed fractional comparison: cast constant to {}",
            get_type_id(*ret_val->type));
        }
        else if (side2_is_retval && is_constant_int2t(*side1))
        {
          *side1 = typecast2tc(ret_val->type, *side1);
          log_debug(
            "contracts",
            "Fixed fractional comparison: cast constant to {}",
            get_type_id(*ret_val->type));
        }
      }
      // Cases 3 & 4: a comparison where one side is the return_value symbol.
      // Two SMT-level sort mismatches can survive Clang's typing once
      // get_decl_ref has corrected the symbol to the function's real return
      // type, both because the global __ESBMC_return_value is declared `int`:
      //
      //   * Bool vs BitVector (issue #4): `__ESBMC_return_value == (c > 0)`
      //     promotes the boolean operand to int. For a bool-returning function
      //     the symbol is bool, leaving one side Bool and the other
      //     (_ BitVec 32) — which Z3 rejects with "Sorts Bool and
      //     (_ BitVec 32) are incompatible" (and core-dumps).
      //
      //   * Integer width (issue #5312): an `int` return value compared with a
      //     wider integer operand after remove_incorrect_casts stripped the
      //     usual-arithmetic-conversion cast (mk_bvsgt requires equal widths).
      else if (side1_is_retval || side2_is_retval)
      {
        expr2tc *rv = side1_is_retval ? side1 : side2;
        expr2tc *other = side1_is_retval ? side2 : side1;

        auto is_int = [](const type2tc &t) {
          return is_signedbv_type(t) || is_unsignedbv_type(t);
        };

        if (is_bool_type((*rv)->type) != is_bool_type((*other)->type))
        {
          // Bool vs BitVector sort mismatch: promote the boolean side to the
          // other operand's type, matching C's usual arithmetic conversions
          // (a bool is widened to int, never the reverse). Casting the boolean
          // up is value-preserving ({0,1} fits any wider type); demoting the
          // other side to bool would collapse values >1 and silently satisfy
          // postconditions such as `__ESBMC_return_value == (x & 3)`.
          expr2tc *boolean = is_bool_type((*rv)->type) ? rv : other;
          expr2tc *wider = (boolean == rv) ? other : rv;
          *boolean = typecast2tc((*wider)->type, *boolean);
          log_debug(
            "contracts",
            "Fixed bool/bitvector comparison: promoted boolean operand to {}",
            get_type_id(*(*wider)->type));
        }
        else if (
          is_int((*rv)->type) && is_int((*other)->type) &&
          (*rv)->type->get_width() != (*other)->type->get_width())
        {
          // Integer width mismatch: widen the narrower side.
          if ((*rv)->type->get_width() < (*other)->type->get_width())
            *rv = typecast2tc((*other)->type, *rv);
          else
            *other = typecast2tc((*rv)->type, *other);
          log_debug(
            "contracts",
            "Fixed integer comparison: widened return_value comparison to "
            "matching width");
        }
      }
    }

    return new_expr;
  }

  // Step 2: Handle logical operators (AND, OR) that may contain comparisons.
  // Recurse into every operand so that nested logical sub-expressions (e.g.
  // and(or(equality(rv, 0), ...), equality(...))) are fully processed.
  if (is_and2t(expr) || is_or2t(expr))
    return transform_operands_if_changed(
      expr, [this, &ret_val](const expr2tc &op) {
        return fix_comparison_types(op, ret_val);
      });

  // Step 3: For all other expressions, return unchanged
  // We don't recursively process arbitrary expression trees to avoid infinite loops
  return expr;
}

expr2tc code_contractst::normalize_fp_add_in_ensures(const expr2tc &expr) const
{
  if (is_nil_expr(expr))
    return expr;

  // NON-RECURSIVE: Only process floating-point add2t expressions
  // Convert regular floating-point addition to IEEE_ADD to match implementation semantics

  if (is_add2t(expr))
  {
    const add2t &add = to_add2t(expr);

    // Only convert if this is a floating-point type
    if (is_fractional_type(add.type))
    {
      // Use default rounding mode symbol (same as implementation)
      expr2tc rounding_mode =
        symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode");

      // Convert to IEEE floating-point addition
      expr2tc new_expr =
        ieee_add2tc(add.type, add.side_1, add.side_2, rounding_mode);

      log_debug(
        "contracts",
        "Normalized floating-point addition to IEEE_ADD in ensures clause");

      return new_expr;
    }
  }

  // For non-add expressions or non-floating-point types, process operands
  // but only one level deep to avoid recursion issues
  if (is_and2t(expr) || is_or2t(expr))
    return transform_operands_if_changed(expr, [this](const expr2tc &op) {
      return normalize_fp_add_in_ensures(op);
    });

  // For comparison expressions, process both sides
  if (is_comp_expr(expr))
  {
    expr2tc new_expr = expr->clone();
    bool changed = false;
    expr2tc *s1 = nullptr, *s2 = nullptr;
    if (get_comparison_sides(new_expr, s1, s2))
    {
      expr2tc norm1 = normalize_fp_add_in_ensures(*s1);
      expr2tc norm2 = normalize_fp_add_in_ensures(*s2);
      if (norm1 != *s1 || norm2 != *s2)
      {
        *s1 = norm1;
        *s2 = norm2;
        changed = true;
      }
    }
    return changed ? new_expr : expr;
  }

  // For typecast expressions, process the inner expression
  if (is_typecast2t(expr))
  {
    const typecast2t &cast = to_typecast2t(expr);
    expr2tc normalized = normalize_fp_add_in_ensures(cast.from);
    if (normalized != cast.from)
    {
      return typecast2tc(cast.type, normalized);
    }
  }

  // For all other expressions, return unchanged
  return expr;
}

// ========== Unified ensures guard normalization ==========

expr2tc code_contractst::normalize_ensures_guard_for_return_value(
  const expr2tc &ensures_clause,
  const expr2tc &ret_val) const
{
  if (is_nil_expr(ensures_clause))
    return ensures_clause;

  expr2tc ensures_guard = ensures_clause;

  // Step 1: Replace __ESBMC_return_value with actual ret_val symbol
  if (!is_nil_expr(ret_val))
  {
    ensures_guard = replace_return_value_in_expr(ensures_guard, ret_val);
  }

  // Step 2: Fix type mismatches in comparison expressions involving return values
  // This removes incorrect casts and adds correct casts for constants
  if (!is_nil_expr(ret_val))
  {
    ensures_guard = fix_comparison_types(ensures_guard, ret_val);
  }

  // Step 3: Normalize floating-point addition to use IEEE semantics (matching implementation)
  // This ensures contracts use IEEE_ADD instead of regular + for floating-point operations
  ensures_guard = normalize_fp_add_in_ensures(ensures_guard);

  return ensures_guard;
}

bool code_contractst::has_contracts(const goto_programt &function_body) const
{
  // __ESBMC_assigns() lowers to an ASSERT marker, not an ASSUME, so the comment
  // scan below cannot see it. An empty frame condition is a whole contract on
  // its own: it states the function writes nothing outside its locals (#6555).
  if (has_empty_assigns_marker(function_body))
    return true;

  // Quick check: scan for contract markers without extracting full clauses
  forall_goto_program_instructions (it, function_body)
  {
    // Check ASSUME instructions for requires/ensures/assigns
    if (it->is_assume())
    {
      std::string comment = id2string(it->location.comment());
      if (
        comment == "contract::requires" || comment == "contract::ensures" ||
        comment == "contract::assigns")
      {
        return true;
      }
    }
    // Also detect assigns-only contracts: __ESBMC_assigns() generates an ASSIGN
    // instruction with a sideeffect of kind assigns_target. A function with
    // only __ESBMC_assigns (no requires/ensures) still has a contract.
    if (it->is_assign())
    {
      const code_assign2t &assign = to_code_assign2t(it->code);
      if (
        is_sideeffect2t(assign.source) &&
        to_sideeffect2t(assign.source).kind ==
          sideeffect2t::allockind::assigns_target)
      {
        return true;
      }
    }
  }
  return false;
}

/// Helper: check if a function name matches any pattern in to_replace.
/// Matching rules:
///   - "*" matches everything
///   - Otherwise, exact match against func_name or its unqualified tail
///     (the part after the last "@"), e.g. pattern "foo" matches "c:@F@foo"
static bool matches_replace_pattern(
  const std::string &func_name,
  const std::set<std::string> &to_replace)
{
  for (const auto &pattern : to_replace)
  {
    if (pattern == "*")
      return true;
    // Exact match on full qualified name (e.g. "c:@F@foo")
    if (func_name == pattern)
      return true;
    // Match unqualified name: extract the part after the last '@'
    // e.g. "c:@F@foo" → "foo", "c:@F@foo#" (C++ free func) → "foo"
    auto pos = func_name.rfind('@');
    if (pos != std::string::npos)
    {
      std::string tail = func_name.substr(pos + 1);
      // Strip trailing '#' used by C++ free function symbols
      if (!tail.empty() && tail.back() == '#')
        tail.pop_back();
      if (tail == pattern)
        return true;
    }
  }
  return false;
}

bool code_contractst::is_annotated_contract_function(
  const symbolt &func_sym) const
{
  // Check if function type has #annotated_contract attribute set
  // This is set in clang_c_convert.cpp when parsing __attribute__((annotate("__ESBMC_contract")))
  return func_sym.get_type().get_bool("#annotated_contract");
}

std::string code_contractst::diagnose_contract_target(
  const std::string &function_name,
  bool for_replace)
{
  if (for_replace)
  {
    // Mirror replace_calls' selection below: a body-available function whose
    // goto key matches the pattern, carrying a contract or the annotation.
    bool any_match = false;
    forall_goto_functions (it, goto_functions)
    {
      if (!it->second.body_available)
        continue;
      if (!matches_replace_pattern(id2string(it->first), {function_name}))
        continue;
      any_match = true;
      symbolt *sym = context.find_symbol(it->first);
      if (
        has_contracts(it->second.body) ||
        (sym && is_annotated_contract_function(*sym)))
      {
        return clause_call_reason(it->second.body);
      }
    }
    return any_match ? "that function declares no contract clauses"
                     : "no function of that name has a body here";
  }

  // Mirror enforce_contracts' gates.
  if (is_compiler_generated(function_name))
    return "that name is compiler-generated";

  const symbolt *sym = find_function_symbol(function_name);
  if (sym == nullptr)
    return "no function of that name, or the name is ambiguous";

  auto it = goto_functions.function_map.find(sym->id);
  if (it == goto_functions.function_map.end() || !it->second.body_available)
    return "that function is declared but has no body here";

  if (!has_contracts(it->second.body) && !is_annotated_contract_function(*sym))
    return "that function declares no contract clauses";

  return clause_call_reason(it->second.body);
}

void code_contractst::replace_calls(const std::set<std::string> &to_replace)
{
  // Build a map of function names to their symbols, bodies, and IDs for quick lookup
  // Key: function name (e.g., "increment")
  // Value: (symbol pointer, body pointer, function ID in goto_functions)
  std::map<std::string, std::tuple<symbolt *, goto_programt *, irep_idt>>
    function_map;

  // Collect all functions that might be called
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;

    symbolt *func_sym = find_function_symbol(id2string(it->first));
    if (func_sym != nullptr)
    {
      // Use the goto_functions key (it->first) as the map key, since that's what
      // get_symbol_name() returns in function calls
      std::string func_key = id2string(it->first);
      function_map[func_key] = {func_sym, &it->second.body, it->first};
      log_debug(
        "contracts",
        "Added function to map: {} (name: {}, id: {})",
        func_key,
        id2string(func_sym->name),
        id2string(it->first));
    }
  }

  // Collect all replaceable functions: those that match the replace pattern
  // AND have contracts (either explicit clauses or annotated).
  std::set<std::string> replaceable_funcs;
  for (const auto &kv : function_map)
  {
    const std::string &func_key = kv.first;
    symbolt *sym = std::get<0>(kv.second);
    goto_programt *body = std::get<1>(kv.second);
    if (!matches_replace_pattern(func_key, to_replace))
      continue;
    if (!has_contracts(*body) && !(sym && is_annotated_contract_function(*sym)))
      continue;
    replaceable_funcs.insert(func_key);
  }

  for (const auto &f : replaceable_funcs)
    log_debug("contracts", "  Replaceable: {}", f);

  log_status(
    "Replacing calls with contracts for {} function(s)",
    replaceable_funcs.size());

  // Track functions to delete after replacement
  std::set<irep_idt> functions_to_delete;

  // Replace ALL calls to replaceable functions, regardless of whether
  // the callee itself calls other replaceable functions.
  // This implements true hierarchical contract replacement:
  //   - Every call to a replaceable function is replaced with its contract
  //     (assert requires → havoc assigns → assume ensures)
  //   - The function definition is removed after replacement
  //   - The caller only sees the contract abstraction, not the function body
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;

    std::vector<goto_programt::targett> calls_to_replace;
    std::vector<std::tuple<symbolt *, goto_programt *, irep_idt>> function_info;

    Forall_goto_program_instructions (i_it, it->second.body)
    {
      if (i_it->is_function_call() && is_code_function_call2t(i_it->code))
      {
        const code_function_call2t &call = to_code_function_call2t(i_it->code);
        if (is_symbol2t(call.function))
        {
          const symbol2t &func_sym = to_symbol2t(call.function);
          std::string called_func = id2string(func_sym.get_symbol_name());

          // Skip compiler-generated functions
          if (is_compiler_generated(called_func))
            continue;

          // Replace calls to ALL replaceable functions
          if (replaceable_funcs.count(called_func) == 0)
            continue;

          auto map_it = function_map.find(called_func);
          if (map_it != function_map.end())
          {
            log_debug("contracts", "Found call to replace: {}", called_func);
            calls_to_replace.push_back(i_it);
            function_info.push_back(map_it->second);
            irep_idt func_id = std::get<2>(map_it->second);
            functions_to_delete.insert(func_id);
          }
        }
      }
    }

    // Replace calls
    log_debug(
      "contracts",
      "Found {} calls to replace in function {}",
      calls_to_replace.size(),
      id2string(it->first));
    for (size_t i = 0; i < calls_to_replace.size(); ++i)
    {
      log_debug(
        "contracts", "Replacing call {} of {}", i + 1, calls_to_replace.size());
      symbolt *func_sym = std::get<0>(function_info[i]);
      goto_programt *func_body = std::get<1>(function_info[i]);
      generate_replacement_at_call(
        *func_sym, *func_body, calls_to_replace[i], it->second.body);
    }
  }

  // Delete all replaced function definitions
  for (const auto &func_id : functions_to_delete)
  {
    auto func_it = goto_functions.function_map.find(func_id);
    if (func_it != goto_functions.function_map.end())
    {
      goto_functions.function_map.erase(func_it);
      log_status("Removed function {} (replaced with contract)", func_id);
    }
    else
    {
      log_warning(
        "Function ID {} not found in function_map for deletion", func_id);
    }
  }

  goto_functions.update();
}

/// The __ESBMC_is_fresh call at \p it, or nullptr if it is not one.
static const code_function_call2t *
as_is_fresh_call(goto_programt::const_targett it)
{
  if (!it->is_function_call() || !is_code_function_call2t(it->code))
    return nullptr;

  const code_function_call2t &c = to_code_function_call2t(it->code);
  if (
    !is_symbol2t(c.function) ||
    !is_fresh_function(to_symbol2t(c.function).thename.as_string()) ||
    c.operands.size() < 2 || is_nil_expr(c.ret) || !is_symbol2t(c.ret))
    return nullptr;

  return &c;
}

/// The position of the parameter \p ptr names, or params.size() if it names
/// none. Recorded before rebinding to actual arguments, which makes two
/// formals bound to one actual indistinguishable from one formal named twice.
static size_t
fresh_param_index(const expr2tc &ptr, const code_typet::argumentst &params)
{
  if (!is_symbol2t(ptr))
    return params.size();

  for (size_t i = 0; i < params.size(); ++i)
    if (params[i].get_identifier() == to_symbol2t(ptr).thename)
      return i;

  return params.size();
}

/// \brief The separations a caller must discharge for the fresh parameters.
///
/// __ESBMC_is_fresh(p, n) says p addresses a *fresh* object, so it is separate
/// from everything else the caller can reach, not merely from the other
/// is_fresh parameters. The enforce harness grants exactly that, by backing p
/// on its own and excluding it from the aliasing introduced for #6551, so
/// every one of those separations has to be discharged at the call. Otherwise
/// a caller passes one object twice, the assumed ensures becomes
/// self-contradictory, and everything after the call is discharged vacuously
/// (the first case in #6542).
/// Static-lifetime objects the contract's clauses name, as addresses. A fresh
/// pointer is separate from everything the caller can reach, so it is separate
/// from these too, and only the ones the contract mentions are worth asserting.
static void collect_clause_globals(
  const expr2tc &e,
  const namespacet &ns,
  std::vector<expr2tc> &out)
{
  if (is_nil_expr(e))
    return;

  if (is_symbol2t(e))
  {
    const symbolt *sym = ns.lookup(to_symbol2t(e).thename);
    if (sym && sym->static_lifetime && !is_code_type(e->type))
    {
      expr2tc addr = address_of2tc(pointer_type2tc(e->type), e);
      if (
        std::find(out.begin(), out.end(), addr) == out.end() &&
        !is_pointer_type(e->type))
        out.push_back(addr);
    }
    return;
  }

  e->foreach_operand(
    [&ns, &out](const expr2tc &op) { collect_clause_globals(op, ns, out); });
}

static std::vector<expr2tc> is_fresh_separations(
  const code_typet::argumentst &params,
  const std::vector<expr2tc> &actual_args,
  const std::set<size_t> &fresh_params,
  const std::vector<expr2tc> &fresh_others,
  const std::vector<expr2tc> &clause_globals)
{
  std::vector<expr2tc> obligations;

  // Every fresh pointer is separate from each global the contract names.
  auto separate_from_globals = [&](const expr2tc &fresh) {
    for (const expr2tc &g : clause_globals)
      obligations.push_back(not2tc(same_object2tc(fresh, g)));
  };

  // Fresh lvalues that are not parameters: a global, `s->p`, `*out`. They owe
  // the same separation, against every pointer argument and against each
  // other, but have no position to key on.
  for (size_t a = 0; a < fresh_others.size(); ++a)
  {
    for (size_t j = 0; j < params.size() && j < actual_args.size(); ++j)
      if (is_pointer_type(migrate_type(params[j].type())))
        obligations.push_back(
          not2tc(same_object2tc(fresh_others[a], actual_args[j])));

    for (size_t b = a + 1; b < fresh_others.size(); ++b)
      obligations.push_back(
        not2tc(same_object2tc(fresh_others[a], fresh_others[b])));

    separate_from_globals(fresh_others[a]);
  }

  for (size_t i : fresh_params)
  {
    if (i >= actual_args.size())
      continue;

    for (size_t j = 0; j < params.size() && j < actual_args.size(); ++j)
    {
      if (j == i || !is_pointer_type(migrate_type(params[j].type())))
        continue;
      // Emit each unordered pair once when both ends are fresh.
      if (fresh_params.count(j) && j < i)
        continue;

      obligations.push_back(
        not2tc(same_object2tc(actual_args[i], actual_args[j])));
    }

    separate_from_globals(actual_args[i]);
  }

  return obligations;
}

/// \brief Lower __ESBMC_is_fresh in a requires clause for a replace site.
///
/// On the assume/enforce side is_fresh is realised by allocation; here the
/// precondition is checked against the caller's argument, so the raw intrinsic
/// -- whose return-value temp is never defined in the caller -- would be
/// asserted against an undefined value and pass vacuously (#6380).
///
/// \param separation Output: obligations the caller must discharge.
/// \return The requires clause with is_fresh temps rewritten.
expr2tc code_contractst::lower_is_fresh_in_requires(
  const symbolt &function_symbol,
  const goto_programt &function_body,
  const std::vector<expr2tc> &actual_args,
  expr2tc requires_clause,
  std::vector<expr2tc> &separation)
{
  if (is_nil_expr(requires_clause) || !function_symbol.get_type().is_code())
    return requires_clause;

  const code_typet::argumentst &params =
    to_code_type(function_symbol.get_type()).arguments();

  std::set<size_t> fresh_params;
  std::vector<expr2tc> fresh_others;
  std::vector<is_fresh_mapping_t> req_is_fresh;

  forall_goto_program_instructions (it, function_body)
  {
    const code_function_call2t *call = as_is_fresh_call(it);
    if (!call)
      continue;

    // Recover the guarded pointer, stripping the void* cast the frontend
    // inserts, then rebind the callee's formal parameter to the actual
    // argument passed at this call site.
    expr2tc ptr = call->operands[0];
    while (is_typecast2t(ptr))
      ptr = to_typecast2t(ptr).from;

    const size_t fresh_param = fresh_param_index(ptr, params);

    // The extent is rebound alongside the pointer: __ESBMC_is_fresh(b, n) with
    // n a parameter has to mean the caller's n, not the callee's symbol.
    expr2tc size = call->operands[1];
    for (size_t i = 0; i < params.size() && i < actual_args.size(); ++i)
    {
      expr2tc param_expr =
        symbol2tc(migrate_type(params[i].type()), params[i].get_identifier());
      ptr = replace_symbol_in_expr(ptr, param_expr, actual_args[i]);
      size = replace_symbol_in_expr(size, param_expr, actual_args[i]);
    }
    if (!is_pointer_type(ptr->type))
      continue;

    // The separation obligation is only owed when the requires clause asserts
    // this is_fresh unconditionally. That test also excludes an ensures-side
    // is_fresh, which describes what the callee produces and asks nothing of
    // the caller. It is made here, while the clause still mentions the temp:
    // the positional test the enforce side uses reads the untransformed body,
    // and by this point the clause assumes have been rewritten.
    if (asserted_unconditionally(
          requires_clause, to_symbol2t(call->ret).thename))
    {
      if (fresh_param != params.size())
        fresh_params.insert(fresh_param);
      else
        // A global, `s->p`, `*out`: the enforce harness allocates that lvalue
        // on its own just as it does a bare parameter, so the separation has
        // to be discharged here too. Keyed on the rebound expression rather
        // than a parameter position, which these have none of.
        fresh_others.push_back(ptr);
    }

    // valid_object() takes a pointer operand (cf. the canonical
    // valid_object2tc(address_of(obj)) in dereference.cpp), so pass the
    // argument pointer directly -- do not dereference it, which would both
    // add a spurious bounds check and (through the frontend's void* cast)
    // build an ill-typed array-select.
    is_fresh_mapping_t m;
    m.temp_var_name = to_symbol2t(call->ret).thename;
    m.ptr_expr = ptr;
    m.size_expr = size;
    req_is_fresh.push_back(m);
  }

  if (!req_is_fresh.empty())
    requires_clause = replace_is_fresh_temps(
      requires_clause, req_is_fresh, /*require_dynamic=*/false);

  std::vector<expr2tc> clause_globals;
  collect_clause_globals(requires_clause, ns, clause_globals);
  collect_clause_globals(
    extract_ensures_from_body(function_body), ns, clause_globals);

  separation = is_fresh_separations(
    params, actual_args, fresh_params, fresh_others, clause_globals);
  return requires_clause;
}

// An array frame target reached through a pointer cannot be assigned in one go:
// dereference refuses to build an array-typed lvalue (dereference.cpp:1103).
// Write the elements instead -- same post-state, more instructions. Returns
// whether the target was of that shape and so is now dealt with.
static bool havoc_pointed_to_array(
  const expr2tc &target,
  const locationt &call_location,
  goto_programt &replacement)
{
  if (!is_array_type(target->type) || !reaches_through_pointer(target))
    return false;

  const array_type2t &at = to_array_type(target->type);
  if (!is_constant_int2t(at.array_size))
  {
    // A symbolic extent has no element count to write out, so the frame cannot
    // be havocked at all. Saying so beats a silent partial havoc.
    log_warning(
      "__ESBMC_assigns: array of unknown size reached through a pointer is "
      "not havocked; the caller keeps its pre-call value");
    return true;
  }

  const BigInt &n = to_constant_int2t(at.array_size).value;
  for (BigInt k = 0; k < n; k += 1)
  {
    expr2tc elem =
      index2tc(at.subtype, target, constant_int2tc(at.array_size->type, k));
    goto_programt::targett e = replacement.add_instruction(ASSIGN);
    e->code = code_assign2tc(elem, gen_nondet(at.subtype));
    e->location = call_location;
    e->location.comment("contract havoc assigns");
  }
  log_debug(
    "contracts",
    "Havoc'd {} elements of a pointed-to array assigns target",
    integer2string(n));
  return true;
}

// The value the removed body would have returned: a fresh unconstrained one,
// which is what `__ESBMC_return_value` is rewritten to.
//
// The caller's own lvalue cannot play that part. The ensures is instantiated
// with the caller's argument expressions, so for `r = f(r)` it would carry `r`
// on both sides and the ASSUME would read `r == r + 1` -- `assume false`, the
// path dies, and a reachable assertion after the call goes unreported (#7009).
// Naming the result separately keeps the arguments at their pre-call values
// for as long as the clause speaks about them; the caller's place is written
// once, after (§4.b).
//
// A discarded result still needs one. The ensures is assumed either way, and
// with nothing to rewrite `__ESBMC_return_value` to it was assumed over a
// symbol no instruction defines.
expr2tc code_contractst::declare_call_result(
  const symbolt &function_symbol,
  const expr2tc &ret_val,
  const locationt &call_location,
  goto_programt &replacement) const
{
  type2tc result_type;
  if (!is_nil_expr(ret_val))
    result_type = ret_val->type;
  else if (function_symbol.get_type().is_code())
    result_type =
      migrate_type(to_code_type(function_symbol.get_type()).return_type());

  // A void function returns nothing for a clause to name.
  if (is_nil_type(result_type) || is_empty_type(result_type))
    return expr2tc();

  // Unlike the three havoc sites, this one does not skip a pointer under
  // --add-symex-value-sets. They leave existing state alone; this one is the
  // result's only definition, and omitting it puts the ensures back over the
  // caller's stale value. A pointer result therefore carries no value set, so
  // a write through it is lost (github_7009_pointer_result_knownbug).
  static size_t counter = 0;
  expr2tc result = declare_local_symbol(
    "__ESBMC_return_value$" + std::to_string(counter++), result_type);

  goto_programt::targett decl = replacement.add_instruction(DECL);
  decl->code = code_decl2tc(result_type, to_symbol2t(result).thename);
  decl->location = call_location;
  decl->location.comment("contract call result declaration");

  goto_programt::targett assign = replacement.add_instruction(ASSIGN);
  assign->code = code_assign2tc(result, gen_nondet(result_type));
  assign->location = call_location;
  assign->location.comment("contract call result");

  return result;
}

// The caller's place takes the result only once the ensures has been assumed:
// until then the arguments the clause was instantiated with must still hold
// their pre-call values, and one of them can be that very place.
static void write_back_call_result(
  const expr2tc &ret_val,
  const expr2tc &call_result,
  const locationt &call_location,
  goto_programt &replacement)
{
  if (is_nil_expr(ret_val) || is_nil_expr(call_result))
    return;

  goto_programt::targett t = replacement.add_instruction(ASSIGN);
  t->code = code_assign2tc(ret_val, call_result);
  t->location = call_location;
  t->location.comment("contract call result written back");
}

void code_contractst::generate_replacement_at_call(
  const symbolt &function_symbol,
  const goto_programt &function_body,
  goto_programt::targett call_instruction,
  goto_programt &caller_body)
{
  // Extract contracts from function body (similar to enforce_contracts)
  expr2tc requires_clause = extract_requires_from_body(function_body);
  expr2tc ensures_clause = extract_ensures_from_body(function_body);
  std::vector<expr2tc> assigns_target_exprs =
    extract_assigns_from_body(function_body);

  // Debug: log extracted clauses
  log_debug(
    "contracts",
    "generate_replacement_at_call: extracted requires clause (nil={})",
    is_nil_expr(requires_clause));
  log_debug(
    "contracts",
    "generate_replacement_at_call: extracted ensures clause (nil={})",
    is_nil_expr(ensures_clause));
  log_debug(
    "contracts",
    "generate_replacement_at_call: extracted {} assigns target expressions",
    assigns_target_exprs.size());

  goto_programt replacement;
  locationt call_location = call_instruction->location;

  // Extract return value and arguments from call instruction
  expr2tc ret_val;
  std::vector<expr2tc> actual_args;
  if (is_code_function_call2t(call_instruction->code))
  {
    const code_function_call2t &call =
      to_code_function_call2t(call_instruction->code);
    ret_val = call.ret;
    actual_args = call.operands;
  }

  // Replace function parameters with actual arguments in contract clauses
  if (function_symbol.get_type().is_code())
  {
    const code_typet &code_type = to_code_type(function_symbol.get_type());
    const code_typet::argumentst &params = code_type.arguments();

    // Build parameter-to-argument mapping
    for (size_t i = 0; i < params.size() && i < actual_args.size(); ++i)
    {
      irep_idt param_id = params[i].get_identifier();
      expr2tc param_expr = symbol2tc(migrate_type(params[i].type()), param_id);
      // Replace parameter symbol with actual argument in requires/ensures
      requires_clause =
        replace_symbol_in_expr(requires_clause, param_expr, actual_args[i]);
      ensures_clause =
        replace_symbol_in_expr(ensures_clause, param_expr, actual_args[i]);

      // Debug: log parameter replacement
      log_debug(
        "contracts",
        "Parameter replacement: {} (arg nil={})",
        id2string(param_id),
        is_nil_expr(actual_args[i]));
    }
  }

  // Debug: log clauses after parameter replacement
  log_debug(
    "contracts",
    "After parameter replacement: requires nil={}, ensures nil={}, function={}",
    is_nil_expr(requires_clause),
    is_nil_expr(ensures_clause),
    id2string(function_symbol.name));

  // 1.b Create call-site snapshots for __ESBMC_old() expressions (if any)
  // This mirrors the snapshot creation in generate_checking_wrapper, but
  // moves the snapshots to the call site instead of a wrapper function.
  std::vector<old_snapshot_t> body_snapshots =
    collect_old_snapshots_from_body(function_body);
  std::vector<old_snapshot_t> callsite_snapshots =
    materialize_old_snapshots_at_callsite(
      body_snapshots, function_symbol, actual_args, replacement, call_location);

  // Lambda function to add contract clause instruction (ASSERT or ASSUME)
  // Used for both requires (ASSERT) and ensures (ASSUME) clauses
  auto add_contract_clause = [&replacement, &call_location](
                               const expr2tc &clause,
                               const goto_program_instruction_typet inst_type,
                               const std::string &comment,
                               const std::string &property = "") {
    if (!should_add_clause_instruction(clause, inst_type))
      return;

    goto_programt::targett t = replacement.add_instruction(inst_type);
    t->guard = clause;
    t->location = call_location;
    t->location.comment(comment);
    if (!property.empty())
      t->location.property(property);
  };

  // 1. Assert requires clause (check precondition at call site)
  //
  // Lower any __ESBMC_is_fresh(p, n) in the requires clause to a concrete,
  // dischargeable predicate before asserting it. On the assume/enforce side
  // is_fresh is realised by allocation (section 0 of generate_checking_wrapper);
  // here the precondition is *checked* against the caller's argument, so the
  // raw is_fresh() intrinsic -- whose return-value temp is never defined in the
  // caller -- would be asserted against an undefined value and fail vacuously
  // (#6380). We build the temp -> pointer mapping from the *actual* argument at
  // this call site (with its real pointee type, so we never dereference through
  // the frontend's void* cast), then replace_is_fresh_temps rewrites the temp
  // to valid_object() on the pointed-to object.
  {
    std::vector<expr2tc> separation;
    requires_clause = lower_is_fresh_in_requires(
      function_symbol, function_body, actual_args, requires_clause, separation);

    for (const expr2tc &obligation : separation)
      add_contract_clause(
        obligation,
        ASSERT,
        "contract requires: __ESBMC_is_fresh argument must not alias another "
        "pointer argument",
        "contract requires");
  }
  add_contract_clause(
    requires_clause, ASSERT, "contract requires", "contract requires");

  // 2. Havoc all potentially modified locations
  // In replace_calls mode, we must havoc everything the function might modify,
  // otherwise the effects cannot propagate from the removed function body.

  bool has_empty_assigns = has_empty_assigns_marker(function_body);

  if (!assigns_target_exprs.empty())
  {
    // 2.1. Precise havoc: Only havoc expressions in assigns clause
    // This implements the key feature for eliminating false counterexamples
    // Now assigns targets are expression trees that need parameter substitution
    for (const expr2tc &target_expr : assigns_target_exprs)
    {
      bool target_is_pointer_param = false;
      expr2tc instantiated_target = instantiate_assigns_target(
        target_expr, function_symbol, actual_args, target_is_pointer_param);

      // `&x` names the place already, and havoc_place resolves it to x.
      // Following the pointer as well would write through x rather than to it,
      // which for an `int **pp` argument means havocking `*x` -- a dereference
      // of whatever the caller's pointer happens to hold.
      const bool names_place_directly = is_address_of2t(instantiated_target);
      instantiated_target = havoc_place(instantiated_target);

      if (target_is_pointer_param && !names_place_directly)
      {
        instantiated_target = havoc_through_pointer(instantiated_target, ns);
        if (is_nil_expr(instantiated_target))
        {
          // The frame the contract named cannot be written, so say so: a
          // dropped target reads exactly like one that was havocked.
          log_warning(
            "__ESBMC_assigns: nothing can be written through the pointer "
            "parameter named at {}; the caller keeps its pre-call value",
            call_location.as_string());
          continue;
        }
      }

      // Skip pointer havoc in value-set mode (consistent with loop invariant).
      // Tested on the place actually written, after widening and after
      // following a pointer parameter: the skip is about assigning a pointer,
      // not about writing through one, and a target the contract named should
      // not vanish because it was reached through a pointer variable.
      if (
        config.options.get_bool_option("add-symex-value-sets") &&
        is_pointer_type(instantiated_target))
        continue;

      if (havoc_pointed_to_array(
            instantiated_target, call_location, replacement))
        continue;

      // One nondet of the target's own type, arrays included. ARRAY_OF ties
      // every element to a single nondet, so a callee that leaves its elements
      // holding different values has no post-state the havoc can express and
      // the ensures ASSUME kills the path (#7010). This is what the loop
      // invariant havoc has always done.
      expr2tc rhs = gen_nondet(instantiated_target->type);

      goto_programt::targett t = replacement.add_instruction(ASSIGN);
      t->code = code_assign2tc(instantiated_target, rhs);
      t->location = call_location;
      t->location.comment("contract havoc assigns");

      log_debug("contracts", "Havoc'd assigns target expression");
    }

    log_debug(
      "contracts",
      "Precise havoc: havoc'd {} expressions from assigns clause",
      assigns_target_exprs.size());
  }
  else if (has_empty_assigns)
  {
    // 2.2. Explicit empty assigns: __ESBMC_assigns(0) was used
    // This means the function is pure (no side effects), so don't havoc anything
    log_debug(
      "contracts",
      "Empty assigns: function is pure (no side effects), no havoc");
  }
  else
  {
    // 2.3. Conservative havoc: No assigns clause, so havoc all globals
    // This is the old behavior - safe but may introduce false positives
    havoc_static_globals(replacement, call_location);

    log_debug(
      "contracts",
      "Conservative havoc: no assigns clause, havoc'd all static globals");
  }

  // 2.4. Havoc memory locations reachable through pointer parameters.
  // When there is no assigns clause (or no empty-assigns marker), we cannot
  // know which locations the function modifies.  Conservatively havoc every
  // non-void, non-function pointer parameter so that the ensures ASSUME does
  // not create a spurious contradiction with the pre-call value.
  //
  // Example:  void increment(int *p) ensures(*p == old(*p)+1)
  //   call site:  int x = 41;  increment(&x);
  // Without this havoc the ASSUME would be  ASSUME(41 == 41+1 = 42), i.e.
  // FALSE, making all subsequent assertions vacuously VERIFICATION SUCCESSFUL.
  if (!has_empty_assigns && function_symbol.get_type().is_code())
  {
    const code_typet &code_type = to_code_type(function_symbol.get_type());
    const code_typet::argumentst &params = code_type.arguments();

    for (size_t i = 0; i < params.size() && i < actual_args.size(); ++i)
    {
      type2tc param_type = migrate_type(params[i].type());

      // Only handle pointer-typed parameters
      if (!is_pointer_type(param_type))
        continue;

      const pointer_type2t &ptr_type = to_pointer_type(param_type);
      // Resolve a struct/union "tag" pointee (symbol_type2t) to its concrete
      // type. A pointer to a named struct migrates with an unresolved symbol
      // subtype; leaving it unresolved makes the dereference2tc / gen_nondet
      // below propagate a symbol type all the way to SMT sort conversion, which
      // aborts with "Unexpected type ID symbol reached SMT conversion" (#6356).
      // The other pointer-havoc paths already apply this ns.follow() step; this
      // one was missing it. ns.follow() is a no-op for non-symbol types.
      type2tc pointee_type = ns.follow(ptr_type.subtype);

      if (skip_pointee_havoc(pointee_type))
        continue;

      // If precise assigns clause was provided it already handled this arg
      if (!assigns_target_exprs.empty())
        continue;

      // Build  *actual_arg  dereference expression
      expr2tc deref = dereference2tc(pointee_type, actual_args[i]);

      // Generate NONDET rhs of the pointee type
      expr2tc rhs = gen_nondet(pointee_type);

      goto_programt::targett t = replacement.add_instruction(ASSIGN);
      t->code = code_assign2tc(deref, rhs);
      t->location = call_location;
      t->location.comment("contract havoc pointer param");

      log_debug(
        "contracts",
        "Havoc'd pointer parameter {} (*p of type {})",
        i,
        get_type_id(*pointee_type));
    }
  }

  // 2.5. Name the value the removed body would have returned.
  expr2tc call_result =
    declare_call_result(function_symbol, ret_val, call_location, replacement);

  // 3. Normalize ensures guard: replace return_value, fix types, normalize floating-point
  expr2tc ensures_guard =
    normalize_ensures_guard_for_return_value(ensures_clause, call_result);

  // 3.b Replace __ESBMC_old() occurrences in ensures using call-site snapshots
  if (!callsite_snapshots.empty() && !is_nil_expr(ensures_guard))
  {
    log_debug(
      "contracts",
      "generate_replacement_at_call: replacing __ESBMC_old expressions in "
      "ensures (before type={})",
      get_type_id(*ensures_guard->type));
    ensures_guard = replace_old_in_expr(ensures_guard, callsite_snapshots);
    log_debug(
      "contracts",
      "generate_replacement_at_call: replaced __ESBMC_old expressions in "
      "ensures (after type={})",
      ensures_guard ? get_type_id(*ensures_guard->type) : "nil");
  }

  // 4. Assume ensures clause (assume postcondition at call site)
  add_contract_clause(
    ensures_guard, ASSUME, "contract ensures", "contract ensures");

  // 4.b The caller's place takes the result only now.
  write_back_call_result(ret_val, call_result, call_location, replacement);

  // Replace the call with the replacement code.
  //
  // The replacement must take over the call's slot so that any GOTO or label
  // targeting the call lands on the replacement. destructive_insert() splices
  // the replacement *before* the call and leaves the call's iterator identity
  // unchanged, so a call at a branch-target position (e.g. the first
  // instruction of an `else` branch) kept incoming jumps pointing at the
  // post-replacement (now SKIP) call — the whole replacement was skipped on
  // that path, silently dropping the contract (#6364). insert_swap() moves the
  // replacement's first instruction into the call's slot (preserving jumps to
  // it) and relocates the original instruction after it; for a call that is not
  // a jump target the two are equivalent.
  size_t replacement_size = replacement.instructions.size();
  log_debug(
    "contracts",
    "Replacement code generated: {} instructions",
    replacement_size);

  if (!replacement.instructions.empty())
  {
    // Turn the original call into a SKIP first so that, once insert_swap moves
    // it after the replacement, it is inert. call_instruction then points at
    // the replacement's first instruction, which is exactly what incoming
    // jumps should reach.
    call_instruction->make_skip();
    caller_body.insert_swap(call_instruction, replacement);
  }
  else
  {
    log_warning(
      "contracts: no replacement code generated for function {}",
      id2string(function_symbol.name));
    call_instruction->make_skip();
  }
}

// ========== Pointer validity assumptions support ==========

/// Report the one extent the harness still assumes without the contract saying
/// so. Deliberately does not suggest __ESBMC_is_fresh: on a struct parameter
/// that would silently discharge __ESBMC_old-based ensures clauses (#6483).
static void warn_assumed_struct_extents(
  const symbolt &func,
  const locationt &location,
  const std::vector<std::string> &params)
{
  if (params.empty())
    return;

  log_warning(
    "{}: {}: struct pointer parameter(s) {} are assumed to address exactly one "
    "element; the contract states no extent for them. Accesses beyond that are "
    "caught, but the first element is admitted unjustified (#6212).",
    location,
    func.name,
    fmt::join(params, ", "));
}

static bool contains_symbol(const expr2tc &e, const irep_idt &name)
{
  if (is_nil_expr(e))
    return false;
  if (is_symbol2t(e) && to_symbol2t(e).thename == name)
    return true;

  bool found = false;
  e->foreach_operand([&found, &name](const expr2tc &op) {
    if (!found)
      found = contains_symbol(op, name);
  });
  return found;
}

/// Whether \p e reads or writes through \p name rather than merely naming it.
static bool dereferences_symbol(const expr2tc &e, const irep_idt &name)
{
  if (is_nil_expr(e))
    return false;
  if ((is_dereference2t(e) || is_index2t(e)) && contains_symbol(e, name))
    return true;

  bool found = false;
  e->foreach_operand([&found, &name](const expr2tc &op) {
    if (!found)
      found = dereferences_symbol(op, name);
  });
  return found;
}

/// Whether the extent of \p param can matter here: it is read or written
/// through, or it escapes into a call that could do either. Contract clauses
/// are still calls in the body at this point, so one scan covers the body and
/// the requires/ensures/assigns clauses alike.
///
/// Answers true whenever the body cannot be inspected or the parameter
/// escapes. A wrong "no" silently drops the warning on an underspecified
/// contract, which is worse than the noise it was reported for (#6511).
bool code_contractst::param_extent_is_observable(
  const symbolt &func,
  const irep_idt &param) const
{
  auto entry = goto_functions.function_map.find(func.id);
  if (entry == goto_functions.function_map.end())
    return true;
  if (!entry->second.body_available)
    return true;

  for (const auto &ins : entry->second.body.instructions)
  {
    if (dereferences_symbol(ins.code, param))
      return true;
    if (dereferences_symbol(ins.guard, param))
      return true;

    if (!ins.is_function_call() || !is_code_function_call2t(ins.code))
      continue;

    for (const expr2tc &arg : to_code_function_call2t(ins.code).operands)
      if (contains_symbol(arg, param))
        return true;
  }
  return false;
}

/// Tell the user why a dereference may fail: the contract states no extent for
/// these parameters, so the harness leaves their extent unconstrained.
static void warn_unstated_extents(
  const symbolt &func,
  const locationt &location,
  const std::vector<std::string> &params)
{
  if (params.empty())
    return;

  log_warning(
    "{}: {}: contract states no extent for pointer parameter(s) {}, so any "
    "dereference will fail its bounds check and the values they point at are "
    "not checked against the assigns clause. State one with "
    "__ESBMC_requires(__ESBMC_is_fresh(<param>, <bytes>)).",
    location,
    func.name,
    fmt::join(params, ", "));
}

void code_contractst::add_pointer_validity_assumptions(
  goto_programt &wrapper,
  const symbolt &func,
  const locationt &location,
  const std::set<irep_idt> &skip_params,
  const std::set<irep_idt> &separated_params,
  std::vector<expr2tc> &allocated_ptrs,
  std::map<irep_idt, param_extentt> &param_extents)
{
  if (!func.get_type().is_code())
    return;

  // Parameters whose extent the contract leaves unstated, collected so the
  // function gets one warning rather than one per parameter.
  std::vector<std::string> nondet_extent, assumed_one_element;

  // Pointer parameters this function backs itself, paired with their pretty
  // names. Each is given its own storage below, which would hand the callee a
  // separation hypothesis no contract clause states, so they are afterwards
  // allowed to alias (issue #6551). Parameters covered by __ESBMC_is_fresh are
  // absent: is_fresh does state separation, so it is theirs to keep.
  std::vector<std::pair<expr2tc, std::string>> aliasable_params;

  for (const auto &param : to_code_type(func.get_type()).arguments())
  {
    if (!param.type().is_pointer())
      continue;

    type2tc param_type = migrate_type(param.type());
    expr2tc p = symbol2tc(param_type, param.get_identifier());

    std::string name = get_pretty_name(id2string(param.get_identifier()));

    // Skip params already allocated by __ESBMC_is_fresh to avoid overwriting
    // the is_fresh allocation, which has the extent the contract asked for.
    // Only those whose is_fresh is asserted unconditionally are also withheld
    // from aliasing; a guarded one states no separation and must not be
    // granted any.
    if (skip_params.count(param.get_identifier()))
    {
      log_debug(
        "contracts",
        "add_pointer_validity_assumptions: skipping {} (allocated by is_fresh)",
        id2string(param.get_identifier()));
      if (!separated_params.count(param.get_identifier()))
        aliasable_params.emplace_back(
          symbol2tc(migrate_type(param.type()), param.get_identifier()), name);
      continue;
    }

    type2tc pointee = ns.follow(to_pointer_type(param_type).subtype);

    // See emit_struct_stack_backing for why structs are carved out.
    // Drop this branch once #6483 is fixed.
    if (is_structure_type(pointee))
    {
      emit_struct_stack_backing(wrapper, p, name, pointee, func, location);
      // Real stack storage, so one element is genuinely dereferenceable even
      // though the contract never asked for it.
      param_extents[param.get_identifier()] = {
        type_byte_size_expr(pointee, &ns), true};
      assumed_one_element.push_back(name);
      aliasable_params.emplace_back(p, name);
      continue;
    }

    param_extents[param.get_identifier()] = {
      emit_pointer_param_malloc(wrapper, p, name, func, location), false};

    allocated_ptrs.push_back(
      retain_allocation_for_free(wrapper, p, name, func, location));
    // The storage is allocated either way; only the advice is withheld, and
    // only when nothing here can observe the extent (#6511).
    if (param_extent_is_observable(func, param.get_identifier()))
      nondet_extent.push_back(name);
    aliasable_params.emplace_back(p, name);
  }

  emit_pointer_param_aliasing(wrapper, func, location, aliasable_params);

  warn_unstated_extents(func, location, nondet_extent);
  warn_assumed_struct_extents(func, location, assumed_one_element);
}

expr2tc code_contractst::retain_allocation_for_free(
  goto_programt &wrapper,
  const expr2tc &allocated,
  const std::string &name,
  const symbolt &func,
  const locationt &location)
{
  std::string backing_name =
    "__ESBMC_harness_backing_" + id2string(func.name) + "_" + name;

  symbolt backing_sym;
  backing_sym.name = backing_name;
  backing_sym.id = backing_name;
  set_symbol_type(backing_sym, allocated->type);
  backing_sym.lvalue = true;
  backing_sym.static_lifetime = false;
  backing_sym.location = location;
  backing_sym.mode = func.mode;
  const irep_idt backing_id = context.move_symbol_to_context(backing_sym)->id;
  expr2tc backing = symbol2tc(allocated->type, backing_id);

  auto backing_decl = wrapper.add_instruction(DECL);
  backing_decl->code = code_decl2tc(allocated->type, backing_id);
  backing_decl->location = location;
  backing_decl->location.comment("harness: retain allocation for free");

  auto backing_assign = wrapper.add_instruction(ASSIGN);
  backing_assign->code = code_assign2tc(backing, allocated);
  backing_assign->location = location;
  backing_assign->location.comment("harness: retain allocation for free");

  return backing;
}

void code_contractst::emit_pointer_param_aliasing(
  goto_programt &wrapper,
  const symbolt &func,
  const locationt &location,
  const std::vector<std::pair<expr2tc, std::string>> &params)
{
  // Parameters that gained an aliasing choice, so the change of behaviour can
  // be explained once rather than left to be inferred from a flag buried in a
  // counterexample.
  std::vector<std::string> may_alias;

  for (size_t j = 1; j < params.size(); ++j)
  {
    const expr2tc &target = params[j].first;

    // Each parameter either takes the value of an earlier one or keeps its own
    // backing. Chaining the choices reaches every partition of the parameters
    // into aliasing groups, because the assignments run in order and so a
    // later parameter reads an earlier one's chosen value.
    //
    // Pointee type is not a barrier. Two parameters of different pointer type
    // can address the same storage, and a contract that needs them apart has
    // to say so like any other. Restricting this to identical types would also
    // put the two sides out of step, since a replace site discharges is_fresh
    // separation against every pointer argument regardless of type.
    expr2tc value = target;
    bool aliased = false;
    for (size_t i = j; i-- > 0;)
    {
      aliased = true;

      // A named flag rather than an inline nondet, so a counterexample says
      // which two parameters the failing trace aliased.
      std::string flag_name = "__ESBMC_harness_alias_" + id2string(func.name) +
                              "_" + params[j].second + "_" + params[i].second;
      symbolt flag_sym;
      flag_sym.name = flag_name;
      flag_sym.id = flag_name;
      set_symbol_type(flag_sym, get_bool_type());
      flag_sym.lvalue = true;
      flag_sym.static_lifetime = false;
      flag_sym.location = location;
      flag_sym.mode = func.mode;
      const irep_idt flag_id = context.move_symbol_to_context(flag_sym)->id;
      expr2tc flag = symbol2tc(get_bool_type(), flag_id);

      auto flag_decl = wrapper.add_instruction(DECL);
      flag_decl->code = code_decl2tc(get_bool_type(), flag_id);
      flag_decl->location = location;
      flag_decl->location.comment("harness: whether two parameters alias");

      auto flag_assign = wrapper.add_instruction(ASSIGN);
      flag_assign->code = code_assign2tc(flag, gen_nondet(get_bool_type()));
      flag_assign->location = location;
      flag_assign->location.comment("harness: aliasing is unconstrained");

      expr2tc source = params[i].first;
      if (source->type != target->type)
        source = typecast2tc(target->type, source);
      value = if2tc(target->type, flag, source, value);
    }

    if (!aliased)
      continue;

    may_alias.push_back(params[j].second);

    auto assign_inst = wrapper.add_instruction(ASSIGN);
    assign_inst->code = code_assign2tc(target, value);
    assign_inst->location = location;
    assign_inst->location.comment(
      "harness: '" + params[j].second +
      "' may alias an earlier parameter, which the contract does not separate");

    log_debug(
      "contracts",
      "emit_pointer_param_aliasing: {} may alias an earlier parameter",
      params[j].second);
  }

  if (may_alias.empty())
    return;

  log_warning(
    "{}: {}: pointer parameter(s) {} may alias another parameter, because the "
    "contract does not state that they are separate. A contract that needs "
    "them separate has to say so, with __ESBMC_requires(<p> != <q>) or "
    "__ESBMC_requires(__ESBMC_is_fresh(<param>, <bytes>)).",
    location,
    func.name,
    fmt::join(may_alias, ", "));
}

void code_contractst::emit_struct_stack_backing(
  goto_programt &wrapper,
  const expr2tc &p,
  const std::string &param_name,
  const type2tc &pointee,
  const symbolt &func,
  const locationt &location)
{
  std::string harness_var_name =
    "__ESBMC_harness_ptr_" + id2string(func.name) + "_" + param_name;

  symbolt harness_sym;
  harness_sym.name = harness_var_name;
  harness_sym.id = harness_var_name;
  set_symbol_type(harness_sym, pointee);
  harness_sym.lvalue = true;
  harness_sym.static_lifetime = false;
  harness_sym.location = location;
  harness_sym.mode = func.mode;
  const irep_idt harness_id = context.move_symbol_to_context(harness_sym)->id;

  goto_programt::targett decl_inst = wrapper.add_instruction(DECL);
  decl_inst->code = code_decl2tc(pointee, harness_id);
  decl_inst->location = location;
  decl_inst->location.comment("harness: stack backing for pointer parameter");

  // ESSENTIAL: symex needs initial SSA versions of all struct fields before
  // any conditional write can create a new version (ITE phi-node).
  expr2tc harness_expr = symbol2tc(pointee, harness_id);
  auto init_inst = wrapper.add_instruction(ASSIGN);
  init_inst->code = code_assign2tc(harness_expr, gen_nondet(pointee));
  init_inst->location = location;
  init_inst->location.comment("harness: initialize stack backing to nondet");

  // p = &harness_var, which is always non-null so no ASSUME is needed.
  auto assign_inst = wrapper.add_instruction(ASSIGN);
  assign_inst->code =
    code_assign2tc(p, address_of2tc(pointer_type2tc(pointee), harness_expr));
  assign_inst->location = location;
  assign_inst->location.comment(
    "harness: point parameter to stack-backed object");

  log_debug(
    "contracts",
    "emit_struct_stack_backing: stack backing for parameter {}",
    id2string(to_symbol2t(p).thename));
}

expr2tc code_contractst::emit_pointer_param_malloc(
  goto_programt &wrapper,
  const expr2tc &p,
  const std::string &param_name,
  const symbolt &func,
  const locationt &location)
{
  // The extent is a named symbol rather than an inline nondet so that the
  // returned expression and the malloc size are the same value, and so that a
  // counterexample shows the extent under a readable name.
  std::string extent_name =
    "__ESBMC_harness_extent_" + id2string(func.name) + "_" + param_name;

  symbolt extent_sym;
  extent_sym.name = extent_name;
  extent_sym.id = extent_name;
  set_symbol_type(extent_sym, size_type2());
  extent_sym.lvalue = true;
  extent_sym.static_lifetime = false;
  extent_sym.location = location;
  extent_sym.mode = func.mode;
  const irep_idt extent_id = context.move_symbol_to_context(extent_sym)->id;
  expr2tc alloc_size = symbol2tc(size_type2(), extent_id);

  auto extent_decl = wrapper.add_instruction(DECL);
  extent_decl->code = code_decl2tc(size_type2(), extent_id);
  extent_decl->location = location;
  extent_decl->location.comment("harness: nondet extent for pointer parameter");

  // A zero extent is admissible and is the point: it is the state that makes
  // an unstated extent fail. The non-null assume further down does not exclude
  // it, because symex models malloc failure as an independent nondet rather
  // than as a function of the requested size.
  auto extent_assign = wrapper.add_instruction(ASSIGN);
  extent_assign->code = code_assign2tc(alloc_size, gen_nondet(size_type2()));
  extent_assign->location = location;
  extent_assign->location.comment("harness: extent is unconstrained");

  auto assign_inst = wrapper.add_instruction(ASSIGN);
  assign_inst->code = code_assign2tc(p, byte_malloc(alloc_size));
  assign_inst->location = location;
  assign_inst->location.comment(
    "harness: allocate backing for '" + param_name +
    "', extent unstated by the contract");

  auto assume_inst = wrapper.add_instruction(ASSUME);
  assume_inst->guard = notequal2tc(p, gen_zero(p->type));
  assume_inst->location = location;
  assume_inst->location.comment(
    "harness: pointer is non-null after allocation");

  log_debug(
    "contracts",
    "emit_pointer_param_malloc: nondet-extent malloc for parameter {}",
    id2string(to_symbol2t(p).thename));

  return alloc_size;
}
