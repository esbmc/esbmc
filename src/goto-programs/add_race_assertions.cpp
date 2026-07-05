#include <goto-programs/add_race_assertions.h>
#include <goto-programs/remove_no_op.h>
#include <goto-programs/rw_set.h>
#include <pointer-analysis/value_sets.h>
#include <util/expr_util.h>
#include <util/migrate.h>
#include <irep2/irep2_guard.h>
#include <util/prefix.h>
#include <util/std_expr.h>
#include <map>
#include <set>

class w_guardst
{
public:
  explicit w_guardst(contextt &_context) : context(_context)
  {
  }

  expr2tc get_w_guard_expr(const rw_sett::entryt &entry)
  {
    return get_guard_symbol_expr(entry.original_expr);
  }

  expr2tc get_assertion(const rw_sett::entryt &entry)
  {
    return not2tc(get_guard_symbol_expr(entry.original_expr));
  }

  void add_initialization(goto_programt &goto_program);

protected:
  contextt &context;

  expr2tc get_guard_symbol_expr(const expr2tc &original_expr)
  {
    // Introduce a RACE_CHECK(&x) marker whose operand is the address of the
    // accessed object; goto_symext::replace_races_check lowers it to the
    // __ESBMC_races_flag slot for that (object, offset) during symbolic
    // execution.
    return races_check2tc(address_of2tc(original_expr->type, original_expr));
  }
};

void w_guardst::add_initialization(goto_programt &goto_program)
{
  // __ESBMC_races_flag is an infinite array of booleans indexed by a unique
  // (pointer object, byte offset) key (see goto_symext::replace_races_check).
  // Declare it and reset every slot to false at program entry.
  type2tc flag_type = array_type2tc(get_bool_type(), expr2tc(), true);
  expr2tc all_false = constant_array_of2tc(flag_type, gen_false_expr());

  symbolt new_symbol;
  new_symbol.id = "c:@F@__ESBMC_races_flag";
  new_symbol.name = new_symbol.id;
  set_symbol_type(new_symbol, flag_type);
  new_symbol.static_lifetime = true;
  new_symbol.set_value(migrate_expr_back(all_false));
  const symbolt &flag = *context.move_symbol_to_context(new_symbol);

  expr2tc flag_symbol;
  migrate_expr(symbol_expr(flag), flag_symbol);

  goto_programt::targett t =
    goto_program.insert(goto_program.instructions.begin());
  t->type = ASSIGN;
  t->code = code_assign2tc(flag_symbol, all_false);
}

// Collect the root symbol of every object whose address is taken in `expr`,
// recording it in `out`. The root is obtained by walking through member, index
// and typecast selectors (`&s.f` -> s, `&a[k]` -> a, `&(T)x` -> x); a
// dereference under address-of (`&*p`) yields no new escapee because it just
// re-forms an existing pointer.
static void
collect_address_taken(const expr2tc &expr, rw_sett::shared_localst &out)
{
  if (is_nil_expr(expr))
    return;

  if (is_address_of2t(expr))
  {
    expr2tc obj = to_address_of2t(expr).ptr_obj;
    while (is_member2t(obj) || is_index2t(obj) || is_typecast2t(obj))
    {
      if (is_member2t(obj))
        obj = to_member2t(obj).source_value;
      else if (is_index2t(obj))
        obj = to_index2t(obj).source_value;
      else
        obj = to_typecast2t(obj).from;
    }
    if (is_symbol2t(obj))
      out.insert(to_symbol2t(obj).thename);
  }

  expr->foreach_operand(
    [&out](const expr2tc &op) { collect_address_taken(op, out); });
}

// A stack local becomes shared between threads when its address is handed to a
// newly spawned thread, i.e. passed as an argument to pthread_create. Only such
// locals need to stay race-eligible when accessed directly by name (issue
// #4424). Collecting *every* address-taken local instead floods large/CUDA
// kernels with race guards, and because each guard inserts yield()
// context-switch points it dilutes context-bounded search and can hide real
// races. Restrict the escape set to the arguments of pthread_create calls.
static void collect_thread_escaped_locals(
  const goto_programt &goto_program,
  rw_sett::shared_localst &out)
{
  forall_goto_program_instructions (i_it, goto_program)
  {
    if (!i_it->is_function_call())
      continue;
    const code_function_call2t &call = to_code_function_call2t(i_it->code);
    if (!is_symbol2t(call.function))
      continue;
    if (
      id2string(to_symbol2t(call.function).thename).find("pthread_create") ==
      std::string::npos)
      continue;
    for (const expr2tc &arg : call.operands)
      collect_address_taken(arg, out);
  }
}

// Collect the identifiers of every function whose *address* is taken in `e`
// (a function used as a value -- assigned to a function pointer, passed as an
// argument, stored in a struct field, etc.). Such a function may later be
// invoked through that pointer from a context this static analysis cannot see,
// so it must never be classified as "always executed atomically".
static void collect_address_taken_functions(
  const expr2tc &e,
  const goto_functionst &goto_functions,
  std::set<irep_idt> &escaped)
{
  if (is_nil_expr(e))
    return;

  if (is_symbol2t(e))
  {
    const irep_idt &name = to_symbol2t(e).thename;
    if (
      goto_functions.function_map.find(name) !=
      goto_functions.function_map.end())
      escaped.insert(name);
  }

  e->foreach_operand([&goto_functions, &escaped](const expr2tc &op) {
    collect_address_taken_functions(op, goto_functions, escaped);
  });
}

// Compute the set of functions that are *always* executed inside an atomic
// region. A function qualifies when every direct call to it is made from an
// atomic context -- lexically inside an atomic_begin/end region, from the body
// of a __VERIFIER_atomic_* function (which goto_convert wraps in an atomic
// region), or from another always-atomic function -- and its address is never
// taken (so it cannot be reached via a function pointer from an unknown,
// possibly non-atomic, context).
//
// Accesses inside such a function can never participate in a data race: at
// every invocation the thread holds the global atomic lock, so no other thread
// runs concurrently. The data-race instrumenter must therefore treat their
// bodies as atomic, exactly as it already does for the lexically-wrapped body
// of a __VERIFIER_atomic_* function. Without this, a shared read in a helper
// reached only from a __VERIFIER_atomic_* function -- e.g. isEmpty() reading
// the shared `top` in SV-COMP's pthread-ext/25_stack, invoked only from
// __VERIFIER_atomic_assert -- is instrumented as a non-atomic access and
// reports a spurious race against a concurrent atomic writer (issues
// #5133-#5135).
//
// Sound by construction: a function is suppressed only when *every* path to it
// is atomic, so no genuinely racy (non-atomic) access is ever hidden. A
// function with a single non-atomic call site, no call site at all (e.g. a
// thread entry point), or an escaping address is left fully instrumented.
//
// Precision limitation (not a soundness issue): a recursive helper reached only
// from atomic contexts is not classified, because the fixpoint requires the
// caller to already be atomic and a function in a recursion cycle never reaches
// that state. Such helpers stay fully instrumented -- they are over-, never
// under-approximated -- so no race is hidden; the original false alarm simply
// persists for that (rare) shape.
static std::set<irep_idt>
compute_always_atomic_functions(const goto_functionst &goto_functions)
{
  std::set<irep_idt> escaped;

  // Per direct call site: the caller, and whether the call instruction is
  // lexically inside an atomic region of that caller.
  struct callsitet
  {
    irep_idt caller;
    bool lexically_atomic;
  };
  std::map<irep_idt, std::vector<callsitet>> callsites;

  forall_goto_functions (f_it, goto_functions)
  {
    const irep_idt &caller = f_it->first;
    // Lexical atomic nesting depth. goto_convert emits balanced
    // atomic_begin/end regions, so the count stays >= 0; an end is only honored
    // while inside a region. Any residual imbalance would merely under-count
    // (treat a call as non-atomic), which is the safe, race-preserving
    // direction.
    int atomic_depth = 0;
    forall_goto_program_instructions (i_it, f_it->second.body)
    {
      if (i_it->is_atomic_begin())
        ++atomic_depth;
      else if (i_it->is_atomic_end() && atomic_depth > 0)
        --atomic_depth;
      else if (i_it->is_function_call())
      {
        const code_function_call2t &call = to_code_function_call2t(i_it->code);
        collect_address_taken_functions(call.ret, goto_functions, escaped);
        for (const expr2tc &arg : call.operands)
          collect_address_taken_functions(arg, goto_functions, escaped);

        if (is_symbol2t(call.function))
          callsites[to_symbol2t(call.function).thename].push_back(
            {caller, atomic_depth > 0});
        else
          // indirect call through a pointer: its target set was already
          // captured as escaped at the pointer's definition site
          collect_address_taken_functions(
            call.function, goto_functions, escaped);
      }
      else
      {
        if (!is_nil_expr(i_it->code))
          collect_address_taken_functions(i_it->code, goto_functions, escaped);
        if (!is_nil_expr(i_it->guard))
          collect_address_taken_functions(i_it->guard, goto_functions, escaped);
      }
    }
  }

  // Seed with the lexically-wrapped __VERIFIER_atomic_* functions, then close
  // under "every call site is atomic" to a fixpoint. The prefix must match the
  // one goto_convert_functions.cpp uses to wrap these bodies in atomic regions.
  std::set<irep_idt> always_atomic;
  forall_goto_functions (f_it, goto_functions)
    if (has_prefix(id2string(f_it->first), "c:@F@__VERIFIER_atomic_"))
      always_atomic.insert(f_it->first);

  bool changed = true;
  while (changed)
  {
    changed = false;
    for (const auto &entry : callsites)
    {
      const irep_idt &callee = entry.first;
      if (always_atomic.count(callee) || escaped.count(callee))
        continue;

      bool all_atomic = true;
      for (const callsitet &cs : entry.second)
        if (!cs.lexically_atomic && !always_atomic.count(cs.caller))
        {
          all_atomic = false;
          break;
        }

      if (all_atomic)
      {
        always_atomic.insert(callee);
        changed = true;
      }
    }
  }

  return always_atomic;
}

void add_race_assertions(
  contextt &context,
  goto_programt &goto_program,
  w_guardst &w_guards,
  const rw_sett::shared_localst &shared_locals,
  bool body_is_atomic)
{
  namespacet ns(context);

  // A function whose every invocation is atomic (see
  // compute_always_atomic_functions) executes its whole body inside the global
  // atomic lock, so -- like the lexically-wrapped body of a __VERIFIER_atomic_*
  // function -- none of its accesses can race. Start the body in the atomic
  // state so its shared reads/writes are handled by the atomic path below and
  // no spurious race assertion is emitted.
  bool is_atomic = body_is_atomic;

  // In data-races-check-only mode every interleaving point is derived from a
  // yield() call, and add_race_assertions only emits those around *non-atomic*
  // shared accesses (the branch below is guarded by !is_atomic). A thread that
  // executes back-to-back atomic regions therefore has no interleaving point
  // between them, so a data race that can only be exposed by another thread
  // interleaving *between* two atomic regions of the same thread is never
  // explored -> false negative (VERIFICATION SUCCESSFUL on a racy program).
  // Track whether the current atomic region touched shared state and, if so,
  // emit a yield() at its boundary (the ATOMIC_END handling below) so those
  // interleavings are generated. See https://github.com/esbmc/esbmc/issues/4423
  const bool races_only =
    config.options.get_bool_option("data-races-check-only");
  bool atomic_region_touched_shared = false;

  // Write flags of the shared writes performed inside the atomic region
  // currently being processed. They are set just before each in-region write
  // but reset only *after* the region's boundary yield (see ATOMIC_END
  // handling below), so the flag stays observable for one interleaving point.
  // See https://github.com/esbmc/esbmc/issues/4975
  std::list<expr2tc> atomic_write_guards;

  Forall_goto_program_instructions (i_it, goto_program)
  {
    goto_programt::instructiont &instruction = *i_it;

    if (instruction.is_atomic_begin())
    {
      is_atomic = true;
      atomic_region_touched_shared = false;
      atomic_write_guards.clear();
    }

    // Accesses inside an atomic region are not individually instrumented, but
    // we still need to know whether the region touched shared state so a
    // context-switch point can be created at its boundary (see note above).
    if (
      races_only && is_atomic && !atomic_region_touched_shared &&
      (instruction.is_assign() || instruction.is_other() ||
       instruction.is_return() || instruction.is_goto() ||
       instruction.is_assert() || instruction.is_function_call() ||
       instruction.is_assume()))
    {
      const expr2tc &probe_expr =
        (instruction.is_goto() || instruction.is_assert()) ? instruction.guard
                                                           : instruction.code;
      if (!is_nil_expr(probe_expr))
      {
        rw_sett probe(ns, i_it, probe_expr);
        if (!probe.entries.empty())
          atomic_region_touched_shared = true;
      }
    }

    // A shared write that occurs *inside* an atomic region (e.g. the body of a
    // __VERIFIER_atomic_* function) is otherwise emitted as a bare assignment:
    // its write flag is never set, so a concurrent *non-atomic* access to the
    // same object in another thread can never observe a writer and the race is
    // missed -> false negative (VERIFICATION SUCCESSFUL on a racy program).
    // See https://github.com/esbmc/esbmc/issues/4975
    //
    // Set the write flag right before the in-region write; the matching reset
    // is deferred until after the atomic-region boundary yield (emitted at
    // ATOMIC_END below), so the flag stays observable across that single
    // interleaving point, mirroring the non-atomic instrumentation. No
    // assertion is added inside the atomic region on purpose: two
    // mutually-exclusive atomic writers must never race with each other, and a
    // check here would turn that race-free pattern into a spurious violation.
    if (races_only && is_atomic && instruction.is_assign())
    {
      rw_sett rw_set(ns, i_it, instruction.code, &shared_locals);

      bool has_write = false;
      forall_rw_set_entries(e_it, rw_set) if (e_it->second.w)
      {
        has_write = true;
        break;
      }

      if (has_write)
      {
        atomic_region_touched_shared = true;

        goto_programt::instructiont original_instruction;
        original_instruction.swap(instruction);

        instruction.make_skip();
        i_it++;

        // set the write flag(s) immediately before the original write -- set
        forall_rw_set_entries(e_it, rw_set) if (e_it->second.w)
        {
          const expr2tc guard_expr = w_guards.get_w_guard_expr(e_it->second);

          goto_programt::targett t = goto_program.insert(i_it);
          t->type = ASSIGN;
          t->code =
            code_assign2tc(guard_expr, e_it->second.get_guard().as_expr());
          t->location = original_instruction.location;
          i_it = ++t;

          // remember it so the reset can be emitted after the boundary yield
          atomic_write_guards.push_back(guard_expr);
        }

        // re-insert the original write
        goto_programt::targett t = goto_program.insert(i_it);
        *t = original_instruction;
        i_it = t; // loop's ++ advances past the original write
      }
    }

    if (
      (instruction.is_assign() || instruction.is_other() ||
       instruction.is_return() || instruction.is_goto() ||
       instruction.is_assert() || instruction.is_function_call() ||
       instruction.is_assume()) &&
      !is_atomic)
    {
      const expr2tc &rw_expr =
        (instruction.is_goto() || instruction.is_assert()) ? instruction.guard
                                                           : instruction.code;

      rw_sett rw_set(ns, i_it, rw_expr, &shared_locals);

      if (rw_set.entries.empty())
        continue;

      goto_programt::instructiont original_instruction;
      original_instruction.swap(instruction);

      instruction.make_skip();
      i_it++;

      {
        goto_programt::targett t = goto_program.insert(i_it);
        t->type = FUNCTION_CALL;
        code_function_callt call;
        call.function() =
          symbol_expr(*context.find_symbol("c:@F@__ESBMC_yield"));

        migrate_expr(call, t->code);
        t->location = original_instruction.location;
        i_it = ++t;
      }

      // Avoid adding too much thread interleaving by using atomic block
      // yield();
      // atomic {Assert tmp_A == 0; tmp_A = 1; A = n;}
      // tmp_A = 0;
      // See https://github.com/esbmc/esbmc/pull/1544
      goto_programt::targett t = goto_program.insert(i_it);
      *t = ATOMIC_BEGIN;
      i_it = ++t;

      // now add assertion for what is read and written
      forall_rw_set_entries(e_it, rw_set)
      {
        goto_programt::targett t = goto_program.insert(i_it);

        t->make_assertion(w_guards.get_assertion(e_it->second));
        t->location = original_instruction.location;
        t->location.user_provided(false);
        t->location.comment(e_it->second.get_comment());
        i_it = ++t;
      }

      // now add assignments for what is written -- set
      forall_rw_set_entries(e_it, rw_set) if (e_it->second.w)
      {
        goto_programt::targett t = goto_program.insert(i_it);

        t->type = ASSIGN;
        t->code = code_assign2tc(
          w_guards.get_w_guard_expr(e_it->second),
          e_it->second.get_guard().as_expr());

        t->location = original_instruction.location;
        i_it = ++t;
      }

      // insert original statement here
      // We need to keep all instructions before the return,
      // so when we process the return we need add the
      // original instruction at the end
      if (!original_instruction.is_return() && !original_instruction.is_goto())
      {
        goto_programt::targett t = goto_program.insert(i_it);

        *t = original_instruction;
        i_it = ++t;
      }

      {
        goto_programt::targett t = goto_program.insert(i_it);

        *t = ATOMIC_END;
        i_it = ++t;
      }

      if (races_only)
      {
        goto_programt::targett t = goto_program.insert(i_it);
        t->type = FUNCTION_CALL;
        code_function_callt call;
        call.function() =
          symbol_expr(*context.find_symbol("c:@F@__ESBMC_yield"));

        migrate_expr(call, t->code);
        t->location = original_instruction.location;
        i_it = ++t;
      }

      // now add assignments for what is written -- reset
      // only write operations need to be reset:
      // tmp_A = 0;
      forall_rw_set_entries(e_it, rw_set) if (e_it->second.w)
      {
        goto_programt::targett t = goto_program.insert(i_it);

        t->type = ASSIGN;
        t->code = code_assign2tc(
          w_guards.get_w_guard_expr(e_it->second), gen_false_expr());

        t->location = original_instruction.location;
        i_it = ++t;
      }

      if (original_instruction.is_return() || original_instruction.is_goto())
      {
        goto_programt::targett t = goto_program.insert(i_it);
        *t = original_instruction;
        i_it = ++t;
      }

      i_it--; // the for loop already counts us up
    }

    if (instruction.is_atomic_end())
    {
      is_atomic = false;

      // Emit an interleaving point right after an atomic region that accessed
      // shared state, so other threads may interleave between consecutive
      // atomic regions. Restricted to regions that touched shared state to
      // avoid growing the interleaving space for purely thread-local atomics.
      // See https://github.com/esbmc/esbmc/issues/4423
      if (races_only && atomic_region_touched_shared)
      {
        atomic_region_touched_shared = false;

        goto_programt::targett after = i_it;
        ++after;
        goto_programt::targett t = goto_program.insert(after);
        t->type = FUNCTION_CALL;
        code_function_callt call;
        call.function() =
          symbol_expr(*context.find_symbol("c:@F@__ESBMC_yield"));
        migrate_expr(call, t->code);
        t->location = instruction.location;
        i_it = t; // loop's ++ advances past the inserted yield

        // Reset the write flags of shared writes performed inside this atomic
        // region, *after* the boundary yield: the flag therefore stayed
        // observable for exactly one interleaving point (issue #4975). The
        // resets are inserted before `after` so they land right after the
        // yield, in order.
        for (const expr2tc &guard_expr : atomic_write_guards)
        {
          goto_programt::targett r = goto_program.insert(after);
          r->type = ASSIGN;
          r->code = code_assign2tc(guard_expr, gen_false_expr());
          r->location = instruction.location;
          i_it = r;
        }
      }

      atomic_write_guards.clear();
    }
  }

  remove_no_op(goto_program);
}

void add_race_assertions(contextt &context, goto_programt &goto_program)
{
  w_guardst w_guards(context);

  rw_sett::shared_localst shared_locals;
  collect_thread_escaped_locals(goto_program, shared_locals);

  // No call-graph context is available for a single program, so no function
  // can be proven always-atomic; instrument every access normally.
  add_race_assertions(
    context, goto_program, w_guards, shared_locals, /*body_is_atomic=*/false);

  w_guards.add_initialization(goto_program);
  goto_program.update();
}

void add_race_assertions(contextt &context, goto_functionst &goto_functions)
{
  w_guardst w_guards(context);

  // An escape analysis must see the whole program: a local declared in one
  // function may have its address taken there and be dereferenced in another
  // thread's entry function. Collect address-taken locals across every body
  // before instrumenting any of them.
  rw_sett::shared_localst shared_locals;
  forall_goto_functions (f_it, goto_functions)
    collect_thread_escaped_locals(f_it->second.body, shared_locals);

  // A function reached only from atomic contexts runs its whole body under the
  // global atomic lock, so its accesses can never race and must be instrumented
  // as atomic (issues #5133-#5135). This is an interprocedural property, so it
  // is computed once over the whole call graph before any body is instrumented.
  const std::set<irep_idt> always_atomic =
    compute_always_atomic_functions(goto_functions);

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->first != goto_functions.main_id())
      add_race_assertions(
        context,
        f_it->second.body,
        w_guards,
        shared_locals,
        always_atomic.count(f_it->first) > 0);

  // get "main"
  goto_functionst::function_mapt::iterator m_it =
    goto_functions.function_map.find(goto_functions.main_id());

  if (m_it != goto_functions.function_map.end())
  {
    goto_programt &main = m_it->second.body;
    w_guards.add_initialization(main);
  }

  goto_functions.update();
}
