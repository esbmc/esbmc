#include <goto-programs/remove_exceptions.h>
#include <goto-programs/exception_typeid.h>
#include <goto-programs/exception_globals.h>
#include <goto-programs/goto_functions.h>

#include <util/namespace.h>
#include <util/context.h>
#include <util/symbol.h>
#include <util/migrate.h>
#include <util/expr_util.h>
#include <util/std_types.h>
#include <util/std_expr.h>
#include <util/message.h>
#include <irep2/irep2_utils.h>

namespace
{
const irep_idt ellipsis_id = "ellipsis";

bool is_pointer_catch(const irep_idt &type)
{
  return type.as_string().find("_ptr") != std::string::npos;
}

/// One catch clause: its static type (or "ellipsis"), the instruction that
/// begins its handler, and — once lowered — the landing instruction the
/// dispatch branches to (which clears the in-flight flag before the body).
struct handlert
{
  irep_idt type;
  goto_programt::targett target;
  goto_programt::targett landing;
};

/// A try region recovered from the positional CATCH push/pop nesting.
struct regiont
{
  std::vector<handlert> handlers;
  goto_programt::targett push;
  goto_programt::targett pop;
  goto_programt::targett after_try;
  int parent;                      // enclosing region index, or -1
  goto_programt::targett dispatch; // landing pad throws/calls branch to
};

/// A throw or a may-throw call, tagged with the region whose try encloses it
/// (-1 = none → propagates straight to the function epilogue).
struct sitet
{
  goto_programt::targett insn;
  int region;
};

/// Lowers throw/catch to guarded control flow over the exception-state
/// globals (issue #5075). Whole-program, all-or-nothing, with correct
/// nested-try propagation (a throw is matched against its enclosing try chain
/// from the innermost outward). This is the only exception path; a program
/// outside the supported subset is reported as unsupported.
class exception_loweringt
{
public:
  exception_loweringt(contextt &context, const namespacet &ns)
    : context(context),
      ns(ns),
      registry(ns),
      thrown(mk_global(exception_globals::thrown_id)),
      type_id(mk_global(exception_globals::typeid_id)),
      value(mk_global(exception_globals::value_id)),
      uncaught_count(mk_global(exception_globals::uncaught_count_id)),
      terminate_reason(mk_global(exception_globals::terminate_reason_id))
  {
  }

  void run(goto_functionst &goto_functions)
  {
    // Turn dynamic_cast<T&> bad_cast intrinsics into ordinary THROWs first, so
    // every later step (may-throw, registry, dispatch) treats them uniformly.
    lower_bad_cast_calls(goto_functions);

    // The handled-exception stack (push/pop_handled, rethrow_current) only
    // matters for std::current_exception — it is what lets current_exception()
    // observe the exception being handled across nested catches. A program that
    // never captures the current exception does not need it, and instrumenting
    // every handler with OM calls would inflate path length for no benefit (and
    // is unsound to even link on a frontend without the C++ exception OM, e.g.
    // Python). So enable the helpers only when the handled stack is actually
    // needed — the program references __ESBMC_current_exception_raw or contains
    // a bare `throw;` — and their bodies are linked; otherwise the lowering uses
    // its inline re-raise fallback, and a plain throw/catch program (no
    // current_exception, no re-raise) pays no handled-stack overhead.
    if (
      program_calls(goto_functions, "c:@F@__ESBMC_current_exception_raw") ||
      program_has_bare_throw(goto_functions))
      for (const char *id :
           {exception_globals::push_handled_id,
            exception_globals::pop_handled_id,
            exception_globals::rethrow_current_id})
      {
        auto it = goto_functions.function_map.find(id);
        if (
          it != goto_functions.function_map.end() && it->second.body_available)
          available_helpers_.insert(id);
      }

    track_uncaught_ = program_reads_uncaught(goto_functions);

    may_throw = compute_may_throw(goto_functions);

    // Single scan: teach the registry any exception hierarchy that lives only in
    // THROW exception_lists (the Python frontend's classes have no `tag-`
    // symbol). Concurrency is sound here: the exception-state globals are
    // thread-local (see create_exception_state_symbols), so each thread raises,
    // catches and clears its own in-flight exception independently.
    for (const auto &fn : goto_functions.function_map)
    {
      if (!fn.second.body_available)
        continue;
      // A dynamic exception specification's allowed types must each have an id
      // so the epilogue membership test (spec_guard) can reference them. The
      // specification is now function metadata, not a THROW_DECL instruction.
      if (
        fn.second.exception_spec.kind ==
        exception_specificationt::kindt::dynamic)
        for (const irep_idt &ty : fn.second.exception_spec.allowed_types)
          registry.register_chain({ty});
      for (const auto &ins : fn.second.body.instructions)
      {
        if (ins.type == THROW)
        {
          const code_cpp_throw2t &t = to_code_cpp_throw2t(ins.code);
          if (!is_nil_expr(t.operand))
            registry.register_chain(t.exception_list);
        }
        else if (ins.type == FUNCTION_CALL)
        {
          const code_function_call2t &c = to_code_function_call2t(ins.code);
          if (is_symbol2t(c.function))
            direct_call_targets.insert(to_symbol2t(c.function).thename);
          collect_thread_entry(c);
        }
      }
    }

    // A thread start routine that is also called directly (by name), or one
    // reached through a computed pointer, cannot get a sound per-function
    // uncaught-escape check (see the member comments): is_entry is a per-
    // function property, but the terminate-on-escape is only correct on the
    // thread-entry edge. Such a program is declined as unsupported (there is no
    // longer an imperative fallback — #5244 removed it), which avoids a missed
    // std::terminate (unresolved routine) or a spurious one (direct call).
    // Declining is sound: it never validates a buggy program. Residual gap
    // (sound, never a missed bug): a routine that is a clean &worker thread
    // entry but is *also* invoked through an indirect call elsewhere keeps
    // is_entry on that path too, so an exception the indirect caller would
    // catch is over-reported as a terminate. That needs call-site-sensitive
    // enforcement at the pthread trampoline, which is blocked until thread-local
    // state propagates across its indirect call.
    if (thread_entry_unresolved)
      return report_unsupported(
        goto_functions, "a thread with an unresolved start routine");
    for (const irep_idt &e : thread_entries)
      if (direct_call_targets.count(e))
        return report_unsupported(
          goto_functions,
          "a thread start routine that is also called directly");

    // Whole-program, all-or-nothing: every participating function must be in
    // the supported subset, otherwise the program is reported as unsupported
    // (there is no longer a partial-lowering or imperative-dispatch fallback).
    for (auto &fn : goto_functions.function_map)
      if (fn.second.body_available && !program_supported(fn.second.body))
        return report_unsupported(
          goto_functions, unsupported_reason_, fn.first);

    // The uncaught-exception check is anchored at the program entry's epilogue.
    // __ESBMC_main is the universal whole-program entry (it runs static init,
    // then calls main / python_user_main), so an escaping exception always
    // reaches it. Without it (e.g. --function isolated verification) an uncaught
    // exception could be silently accepted, so report it as unsupported.
    bool has_entry = false;
    for (const auto &fn : goto_functions.function_map)
      if (fn.first == "__ESBMC_main")
        has_entry = true;
    if (!has_entry)
      return report_unsupported(
        goto_functions, "no whole-program entry (--function verification)");

    for (auto &fn : goto_functions.function_map)
      if (fn.second.body_available)
        lower_ip(fn.second, fn.first);

    // The globals are created after the frontend builds __ESBMC_main's static
    // initialisation, so seed them at entry; otherwise symex does not track the
    // cross-frame writes that carry an exception out of a callee.
    auto entry = goto_functions.function_map.find("__ESBMC_main");
    if (
      entry != goto_functions.function_map.end() &&
      entry->second.body_available)
      init_globals(entry->second.body);
  }

  /// True if any function still has an exception construct to lower (a throw, a
  /// catch, or a dynamic exception specification). Drives the whole-pass no-op
  /// gate: a surviving THROW_DECL must still be lowered, so it counts here.
  static bool program_uses_exceptions(const goto_functionst &gf)
  {
    for (const auto &fn : gf.function_map)
      if (fn.second.body_available)
      {
        // A restrictive specification (noexcept / throw(...)) gets epilogue
        // enforcement, so the pass must run when one is present. (lower_ip
        // still no-ops on a body that cannot reach the epilogue with an
        // exception in flight — no throws, calls or catches.)
        if (fn.second.exception_spec.is_restrictive())
          return true;
        for (const auto &ins : fn.second.body.instructions)
          if (ins.type == THROW || ins.type == CATCH)
            return true;
      }
    return false;
  }

  /// True if any function directly calls the named function symbol.
  static bool program_calls(const goto_functionst &gf, const irep_idt &callee)
  {
    for (const auto &fn : gf.function_map)
      if (fn.second.body_available)
        for (const auto &ins : fn.second.body.instructions)
          if (ins.type == FUNCTION_CALL)
          {
            const code_function_call2t &c = to_code_function_call2t(ins.code);
            if (
              is_symbol2t(c.function) &&
              to_symbol2t(c.function).thename == callee)
              return true;
          }
    return false;
  }

  /// True if any function contains a bare `throw;` (a THROW with no operand),
  /// which re-raises the exception being handled and so needs the handled stack.
  static bool program_has_bare_throw(const goto_functionst &gf)
  {
    for (const auto &fn : gf.function_map)
      if (fn.second.body_available)
        for (const auto &ins : fn.second.body.instructions)
          if (
            ins.type == THROW &&
            is_nil_expr(to_code_cpp_throw2t(ins.code).operand))
            return true;
    return false;
  }

  /// True if any function calls std::uncaught_exception or uncaught_exceptions
  /// (matched by name to stay robust to overload mangling) — the only readers of
  /// the uncaught count, so the count is maintained only when one is present.
  static bool program_reads_uncaught(const goto_functionst &gf)
  {
    for (const auto &fn : gf.function_map)
      if (fn.second.body_available)
        for (const auto &ins : fn.second.body.instructions)
          if (ins.type == FUNCTION_CALL)
          {
            const code_function_call2t &c = to_code_function_call2t(ins.code);
            if (
              is_symbol2t(c.function) &&
              id2string(to_symbol2t(c.function).thename)
                  .find("uncaught_exception") != std::string::npos)
              return true;
          }
    return false;
  }

  /// The pass cannot lower this program. Lowering is the only exception path
  /// (the legacy imperative path was removed once the lowered subset covered
  /// the corpus, #5075), so an exception-using program the pass declines is
  /// reported as a hard error rather than silently miscompiled. A program with
  /// no exception construct is silent: the pass is a no-op for it, so a
  /// decline that does not touch exceptions (e.g. a concurrent but
  /// exception-free program) returns without complaint.
  void report_unsupported(
    const goto_functionst &gf,
    const std::string &why,
    const irep_idt &fn = irep_idt())
  {
    if (!program_uses_exceptions(gf))
      return;
    // Throw the ESBMC fatal-error idiom (a std::string caught by
    // process_goto_program), which logs the message and stops verification
    // cleanly — rather than abort()/SIGABRT, which reads as an internal crash.
    std::string msg = "exception lowering: cannot lower " + why;
    if (!fn.empty())
      msg += " in '" + id2string(fn) + "'";
    throw msg;
  }

  void init_globals(goto_programt &body)
  {
    auto front = body.instructions.begin();
    auto seed = [&](const expr2tc &g) {
      expr2tc zero;
      migrate_expr(gen_zero(migrate_type_back(g->type)), zero);
      auto n = body.insert(front);
      n->make_assignment();
      n->code = code_assign2tc(g, zero);
      if (front != body.instructions.end())
      {
        n->location = front->location;
        n->function = front->function;
      }
    };
    seed(thrown);
    seed(type_id);
    seed(value);
    seed(terminate_reason);
    body.update();
  }

private:
  contextt &context;
  const namespacet &ns;
  exception_typeidt registry;
  expr2tc thrown, type_id, value, uncaught_count, terminate_reason;
  std::set<irep_idt> may_throw;
  // Handled-stack OM helpers whose body is linked (set in run()); a call is
  // emitted only for these, else the lowering uses its inline fallback.
  std::set<irep_idt> available_helpers_;
  // Whether to maintain $esbmc_exc_uncaught_count (set in run()): only when the
  // program reads it via std::uncaught_exception(s); otherwise the ++/-- at
  // throws and handlers is pure overhead, so it is skipped (pay-per-use).
  bool track_uncaught_ = false;
  // Functions passed as the start routine to pthread_create: each is a thread
  // entry, so an exception escaping it is uncaught for that thread (terminate).
  std::set<irep_idt> thread_entries;
  // Symbols that appear as a direct (by-name) call target anywhere. A thread
  // entry that is *also* called directly cannot be marked is_entry soundly (the
  // terminate at its epilogue would wrongly fire on the direct-call path), so
  // such a program is declined; this records the candidates for that check.
  std::set<irep_idt> direct_call_targets;
  // A pthread_create whose start routine is a computed/unresolved pointer: its
  // thread cannot get the uncaught-escape check, so decline rather than miss
  // it silently.
  bool thread_entry_unresolved = false;
  unsigned storage_counter = 0;

  expr2tc mk_global(const char *id)
  {
    const symbolt *s = ns.lookup(irep_idt(id));
    assert(s && "exception-state globals must be created before lowering");
    return symbol2tc(migrate_type(s->get_type()), s->id);
  }

  /// Assignment code stepping the per-thread uncaught-exception count: +1 at a
  /// throw/rethrow (the exception becomes uncaught) and -1 when a handler is
  /// entered (it stops being uncaught). Backs std::uncaught_exception(s),
  /// [except.uncaught].
  expr2tc adjust_uncaught(int delta)
  {
    const type2tc &t = uncaught_count->type;
    expr2tc one = constant_int2tc(t, BigInt(1));
    expr2tc rhs = delta > 0 ? add2tc(t, uncaught_count, one)
                            : sub2tc(t, uncaught_count, one);
    return code_assign2tc(uncaught_count, rhs);
  }

  expr2tc make_c_helper_call(const irep_idt &id)
  {
    // Only emit the call when the helper's body is actually linked (see
    // available_helpers_); otherwise the caller falls back to inline lowering.
    if (!available_helpers_.count(id))
      return expr2tc();
    const symbolt *h = ns.lookup(id);
    if (!h)
      return expr2tc();
    code_function_callt fc;
    fc.function() = symbol_exprt(h->id, h->get_type());
    expr2tc call;
    migrate_expr(fc, call);
    return call;
  }

  /// If @p call is a pthread_create, record its start-routine argument (the 3rd)
  /// as a thread entry, so lower_ip enforces the uncaught-escape terminate at
  /// that function's epilogue. The argument is `&worker`, possibly under
  /// typecasts; peel them to the underlying function symbol. A computed
  /// (unresolvable) routine sets thread_entry_unresolved so run() declines the
  /// program rather than silently miss its uncaught-escape check.
  void collect_thread_entry(const code_function_call2t &call)
  {
    if (
      !is_symbol2t(call.function) ||
      id2string(to_symbol2t(call.function).thename).find("pthread_create") ==
        std::string::npos ||
      call.operands.size() < 3)
      return;

    expr2tc rtn = call.operands[2];
    while (is_typecast2t(rtn))
      rtn = to_typecast2t(rtn).from;
    if (is_address_of2t(rtn))
      rtn = to_address_of2t(rtn).ptr_obj;
    // Only a direct &function resolves to a code-typed symbol; a function-
    // pointer variable (is_symbol2t but pointer-typed) is a computed routine.
    if (is_symbol2t(rtn) && is_code_type(rtn->type))
      thread_entries.insert(to_symbol2t(rtn).thename);
    else
      thread_entry_unresolved = true;
  }

  /// A fresh static-lifetime slot to hold a copy of a thrown object, so it
  /// outlives the throwing frame (the imperative path's symex_throw::thrown_obj
  /// analogue). One slot per throw site; a single exception is in flight at a
  /// time, so reuse across non-recursive throws is safe.
  expr2tc make_exception_storage(const type2tc &obj_type)
  {
    const std::string id =
      "c:@__ESBMC_exc_obj$" + std::to_string(storage_counter++);
    symbolt sym;
    sym.id = id;
    sym.name = id;
    sym.mode = "C";
    typet t1 = migrate_type_back(obj_type);
    sym.set_type(t1);
    sym.set_value(gen_zero(t1));
    sym.lvalue = true;
    sym.static_lifetime = true;
    sym.file_local = false;
    sym.is_thread_local = true;
    context.move_symbol_to_context(sym);
    return symbol2tc(obj_type, irep_idt(id));
  }

  // ---- bad_cast lowering -------------------------------------------------

  /// THROW code for a synthesized std::bad_cast (operand + exception_list =
  /// dynamic type and its bases), or nil if the <typeinfo> model's class is not
  /// in the symbol table. (This is the lowered replacement for the former
  /// goto_symext::symex_throw_bad_cast synthesis.) The tag is elaborated
  /// ("class std::bad_cast") on newer Clang/LLVM, un-elaborated on older, so
  /// try both. Built once and reused across call sites (a single exception is
  /// in flight at a time).
  expr2tc build_bad_cast_throw()
  {
    const symbolt *sym = ns.lookup("tag-std::bad_cast");
    if (!sym)
      sym = ns.lookup("tag-class std::bad_cast");
    if (!sym)
      return expr2tc();

    std::vector<irep_idt> exception_list;
    const std::string tag = id2string(sym->id);
    exception_list.emplace_back(tag.substr(4)); // strip "tag-"
    if (sym->get_type().id() == "struct")
    {
      const struct_typet &st = to_struct_type(sym->get_type());
      const exprt &bases = static_cast<const exprt &>(st.find("bases"));
      if (bases.is_not_nil())
        for (const auto &base : bases.get_sub())
          exception_list.emplace_back(id2string(base.id()).substr(4));
    }

    // A static slot holds the thrown bad_cast object (its state is irrelevant —
    // only its type identity and address matter to the handler).
    expr2tc operand = make_exception_storage(migrate_symbol_type(*sym));
    return code_cpp_throw2tc(operand, exception_list);
  }

  /// Replace each __ESBMC_throw_bad_cast() call — the bodyless intrinsic a
  /// failing dynamic_cast<T&> lowers to — so the rest of the pass never sees the
  /// call. The frontend guards it with the cast-failed condition
  /// (`if (!vptr_matches) __ESBMC_throw_bad_cast();`), so the replacement
  /// inherits that guard. When std::bad_cast is resolvable it becomes the
  /// equivalent THROW (lowered like any other throw, so a surrounding handler
  /// can catch it). When it is not (no <typeinfo>), there is no exception object
  /// to throw, so it becomes an ASSERT(false): a failing reference cast with no
  /// RTTI model is std::terminate, hence a verification error at that point — a
  /// cast that always succeeds leaves the assertion unreachable.
  void lower_bad_cast_calls(goto_functionst &goto_functions)
  {
    expr2tc bad_cast_throw = build_bad_cast_throw();

    for (auto &fn : goto_functions.function_map)
    {
      if (!fn.second.body_available)
        continue;
      for (auto &ins : fn.second.body.instructions)
      {
        if (ins.type != FUNCTION_CALL)
          continue;
        const code_function_call2t &c = to_code_function_call2t(ins.code);
        if (
          !is_symbol2t(c.function) ||
          to_symbol2t(c.function).thename != "c:@F@__ESBMC_throw_bad_cast")
          continue;

        if (!is_nil_expr(bad_cast_throw))
        {
          // make_throw() preserves location/function (clear() leaves them).
          ins.make_throw();
          ins.code = bad_cast_throw;
        }
        else
        {
          ins.make_assertion(gen_false_expr());
          ins.location.property("assertion");
          ins.location.comment("dynamic_cast<T&> failed: std::bad_cast");
        }
      }
    }
  }

  // ---- may-throw analysis ------------------------------------------------

  static std::set<irep_idt> compute_may_throw(const goto_functionst &gf)
  {
    std::set<irep_idt> may;
    std::map<irep_idt, std::set<irep_idt>> callees;
    for (const auto &fn : gf.function_map)
    {
      if (!fn.second.body_available)
        continue;
      for (const auto &ins : fn.second.body.instructions)
      {
        if (ins.type == THROW)
          may.insert(fn.first);
        else if (ins.type == FUNCTION_CALL)
        {
          const code_function_call2t &c = to_code_function_call2t(ins.code);
          if (is_symbol2t(c.function))
          {
            const irep_idt callee = to_symbol2t(c.function).thename;
            if (
              id2string(callee).find("__ESBMC_rethrow_exception_raw") !=
              std::string::npos)
              may.insert(fn.first);
            else
              callees[fn.first].insert(callee);
          }
          else
            may.insert(fn.first); // indirect call: callee may throw
        }
      }
    }
    for (bool changed = true; changed;)
    {
      changed = false;
      for (const auto &[caller, cs] : callees)
        if (!may.count(caller))
          for (const irep_idt &callee : cs)
            if (may.count(callee))
            {
              may.insert(caller);
              changed = true;
              break;
            }
    }
    return may;
  }

  // ---- per-function lowering ---------------------------------------------

  /// Whole-program gate: is every exception construct in this function within
  /// the supported subset (reference catches of registered class types, throws
  /// of registered objects or rethrows, balanced CATCH push/pop nesting)?
  /// Read-only.
  /// Set when program_supported declines, naming the construct for the
  /// unsupported-program diagnostic (report_unsupported).
  std::string unsupported_reason_;

  bool unsupported(const char *why)
  {
    unsupported_reason_ = why;
    return false;
  }

  bool program_supported(goto_programt &body)
  {
    const auto end = body.instructions.end();
    int depth = 0;
    for (auto it = body.instructions.begin(); it != end; ++it)
    {
      // Exception specifications (noexcept / throw(T...) / throw()) are function
      // metadata enforced at the epilogue (see lower_ip), so they never force
      // fallback here; this scan only rejects unsupported catch/throw shapes.
      if (it->type == CATCH && !it->targets.empty())
      {
        const code_cpp_catch2t &c = to_code_cpp_catch2t(it->code);
        if (c.exception_list.size() != it->targets.size())
          return unsupported("a malformed catch clause");
        auto type_it = c.exception_list.begin();
        for (auto tgt : it->targets)
        {
          const irep_idt &ty = *type_it++;
          if (ty == ellipsis_id)
            continue;
          // Reference, value and pointer (`_ptr`) catches are all bindable, as
          // long as the handler is the expected DECL + binding-ASSIGN pair. The
          // catch type need not be registered: an unregistered type is one no
          // throw can match, so its guard is simply false (a dead handler).
          if (!is_code_decl2t(tgt->code))
            return unsupported("an unsupported handler shape");
          auto bind = std::next(tgt);
          if (bind == end || !is_code_assign2t(bind->code))
            return unsupported(
              "a value catch without a copy binding (e.g. catching "
              "std::bad_exception by value)");
        }
        ++depth;
      }
      else if (it->type == CATCH)
      {
        // pop: build_dispatch anchors the region's dispatch block on the
        // skip-handlers GOTO that normally follows. When every handler is empty
        // (e.g. `catch (...) {}`) the frontend elides that GOTO since it would
        // jump to the pop's own fall-through; insert_elided_skip_gotos restores
        // it before lowering, so a pop not followed by a GOTO is supported. An
        // unmatched pop (depth == 0) or trailing pop (nx == end) is malformed.
        if (depth == 0 || std::next(it) == end)
          return unsupported("an unmatched or trailing CATCH pop");
        --depth;
      }
      else if (it->type == THROW)
      {
        const code_cpp_throw2t &t = to_code_cpp_throw2t(it->code);
        if (is_nil_expr(t.operand))
          continue; // rethrow
        if (
          t.exception_list.empty() ||
          !registry.is_registered(t.exception_list.front()))
          return unsupported("a throw of an unsupported type");
      }
      // A __ESBMC_throw_bad_cast() call cannot reach here: lower_bad_cast_calls
      // (run before this check) rewrites every such call to a THROW or an
      // ASSERT(false), so a failing dynamic_cast<T&> is always lowered.
    }
    // depth > 0 means some try's empty-CATCH pop was pruned by remove_unreachable
    // (its body cannot complete normally). Such unclosed pushes are accepted:
    // lower_ip's rebalance_removed_pops re-inserts the synthetic pop before
    // lowering. depth < 0 (an unmatched pop) is already rejected above.
    return depth >= 0;
  }

  /// `remove_unreachable` runs before this pass and prunes the empty CATCH pop
  /// and skip-GOTO of a try whose body cannot complete normally — the common
  /// Python idiom `try: <raises>; assert False; except E: ...`, where the model-
  /// or user-raised throw makes the fall-through dead. That leaves the CATCH
  /// push unbalanced, which the region recovery cannot pair. Re-insert a
  /// synthetic pop + skip-GOTO immediately before each unclosed push's first
  /// handler, restoring the balanced shape collect()/build_dispatch expect. The
  /// skip-GOTO sits on the (now infeasible, hence pruned) normal-completion
  /// path, so its target is immaterial; the function's END_FUNCTION is a valid
  /// choice. Called only when the whole program is supported, so it never
  /// perturbs a body that falls back to the imperative path.
  void rebalance_removed_pops(goto_programt &body)
  {
    const auto end = body.instructions.end();
    std::vector<goto_programt::targett> open;
    for (auto it = body.instructions.begin(); it != end; ++it)
    {
      if (it->type != CATCH)
        continue;
      if (!it->targets.empty())
        open.push_back(it);
      else if (!open.empty())
        open.pop_back();
    }
    if (open.empty())
      return;

    auto end_fn = std::prev(end); // END_FUNCTION
    for (auto push : open)
    {
      // The try body ends at the first handler in program order; that is where
      // the removed pop belonged.
      goto_programt::targett first = end;
      for (auto it = std::next(push); it != end; ++it)
        if (
          std::find(push->targets.begin(), push->targets.end(), it) !=
          push->targets.end())
        {
          first = it;
          break;
        }
      assert(first != end && "CATCH push with no reachable handler");

      auto pop = body.insert(first); // synthetic empty-CATCH pop
      pop->type = CATCH;
      pop->location = push->location;
      pop->function = push->function;
      auto skip = body.insert(first); // skip-handlers GOTO (dead path)
      skip->make_goto(end_fn);
      skip->location = push->location;
      skip->function = push->function;
    }
    body.update();
  }

  /// A try whose handlers are all empty (e.g. `catch (...) {}`) has its
  /// skip-handlers GOTO elided by the frontend: the jump would target the
  /// instruction immediately after the pop, so it is a no-op and is dropped,
  /// leaving the pop directly followed by the first handler. build_dispatch
  /// anchors each region's dispatch block on that GOTO (placed just after it so
  /// normal completion bypasses dispatch), so restore the canonical shape by
  /// re-inserting `GOTO <next>` after any pop that lacks it. The jump targets
  /// the pop's own fall-through, so it is behaviour-neutral on every path.
  void insert_elided_skip_gotos(goto_programt &body)
  {
    const auto end = body.instructions.end();
    for (auto it = body.instructions.begin(); it != end; ++it)
    {
      if (it->type != CATCH || !it->targets.empty())
        continue; // not a pop
      auto nx = std::next(it);
      if (nx == end || nx->type == GOTO)
        continue; // skip-handlers GOTO already present
      auto skip = body.insert(nx);
      skip->make_goto(nx);
      skip->location = it->location;
      skip->function = it->function;
    }
  }

  /// Recover the region tree, throw sites, and may-throw call sites (each with
  /// its enclosing try region).
  void collect(
    goto_programt &body,
    std::vector<regiont> &regions,
    std::vector<sitet> &throws,
    std::vector<sitet> &calls)
  {
    std::vector<int> open;
    for (auto it = body.instructions.begin(); it != body.instructions.end();
         ++it)
    {
      const int here = open.empty() ? -1 : open.back();
      switch (it->type)
      {
      case CATCH:
        if (!it->targets.empty())
        {
          const code_cpp_catch2t &c = to_code_cpp_catch2t(it->code);
          regiont r;
          r.push = it;
          r.parent = here;
          auto type_it = c.exception_list.begin();
          for (auto tgt : it->targets)
            r.handlers.push_back({*type_it++, tgt, {}});
          regions.push_back(std::move(r));
          open.push_back((int)regions.size() - 1);
        }
        else
        {
          regions[open.back()].pop = it;
          auto skip = std::next(it);
          assert(skip != body.instructions.end() && skip->type == GOTO);
          assert(!skip->targets.empty());
          regions[open.back()].after_try = *skip->targets.begin();
          open.pop_back();
        }
        break;

      case THROW:
        throws.push_back({it, here});
        break;

      case FUNCTION_CALL:
      {
        // Guard after any call that may leave an exception in flight: a known
        // may-throw callee, or an indirect call whose target we cannot resolve
        // (so it must be assumed to throw — dropping the guard would silently
        // lose the exception).
        const code_function_call2t &c = to_code_function_call2t(it->code);
        if (
          !is_symbol2t(c.function) ||
          may_throw.count(to_symbol2t(c.function).thename))
          calls.push_back({it, here});
        break;
      }

      default:
        break;
      }
    }
  }

  /// The landing a propagating exception in @p region branches to: that
  /// region's dispatch block, or the function epilogue at the outermost level.
  goto_programt::targett target_for(
    const std::vector<regiont> &regions,
    int region,
    goto_programt::targett epilogue)
  {
    return region == -1 ? epilogue : regions[region].dispatch;
  }

  void lower_ip(goto_functiont &function, const irep_idt &fn_id)
  {
    goto_programt &body = function.body;
    rebalance_removed_pops(body);
    insert_elided_skip_gotos(body);

    std::vector<regiont> regions;
    std::vector<sitet> throws, calls;
    collect(body, regions, throws, calls);

    // The function's exception specification is now metadata on the function
    // (replacing the old THROW_DECL instructions): non_throwing for noexcept,
    // dynamic for throw(T...) / throw() with `allowed_types` the listed types.
    const exception_specificationt &spec = function.exception_spec;

    if (regions.empty() && throws.empty() && calls.empty())
      return;

    // A skip just before END_FUNCTION: a propagating exception lands here and
    // falls through to the return, carrying __ESBMC_exc_thrown to the caller.
    auto epilogue = make_epilogue(body);

    for (regiont &r : regions)
      for (handlert &h : r.handlers)
        h.landing = make_landing(body, h);

    build_dispatch(body, regions, epilogue);
    build_handled_stack(body, regions);

    for (const sitet &s : throws)
      wire_throw(body, s.insn, target_for(regions, s.region, epilogue));
    for (const sitet &s : calls)
      wire_call(body, s.insn, target_for(regions, s.region, epilogue));

    for (regiont &r : regions)
    {
      r.push->make_skip();
      r.pop->make_skip();
    }

    // Enforce the function's exception specification at the epilogue, where an
    // exception still in flight is about to escape. A locally-caught throw
    // clears `thrown`, so this fires only on a genuine escape.
    //  - main / __ESBMC_main: any escape is uncaught (covers static-init throws
    //    from global constructors) → std::terminate.
    //  - a thread start routine (passed to pthread_create): an exception
    //    escaping it is uncaught for that thread → std::terminate
    //    ([except.terminate]). Checked at the routine's own epilogue, where it
    //    set `thrown`, rather than relying on cross-frame propagation out
    //    through the pthread trampoline.
    //  - noexcept: an escape calls std::terminate ([except.spec]).
    //  - throw(T...) / throw(): a disallowed escape runs std::unexpected and is
    //    re-checked (build_dynamic_spec_check), matching the imperative path
    //    (which models the unexpected-handler dispatch and its recovery).
    // A locally-caught throw clears `thrown`, so these fire only on a genuine
    // escape (terminate iff an exception is still in flight). Match `main` by
    // its bare id too (extern "C"/clang-c entry), mirroring
    // goto_convert_functions.cpp.
    const bool is_entry =
      fn_id == "c:@F@main" || id2string(fn_id).rfind("c:@F@main#", 0) == 0 ||
      fn_id == "__ESBMC_main" || thread_entries.count(fn_id);

    if (spec.kind == exception_specificationt::kindt::dynamic)
      build_dynamic_spec_check(body, epilogue, spec.allowed_types, is_entry);
    else if (
      is_entry || spec.kind == exception_specificationt::kindt::non_throwing)
      emit_terminate(
        body,
        std::next(epilogue),
        equality2tc(thrown, gen_false_expr()),
        epilogue->location,
        epilogue->function,
        is_entry ? exception_globals::terminate_reason_uncaught
                 : exception_globals::terminate_reason_noexcept);

    body.update();
  }

  goto_programt::targett make_epilogue(goto_programt &body)
  {
    // A function body always terminates with END_FUNCTION as its last
    // instruction, so there is no need to scan for it.
    auto last = std::prev(body.instructions.end());
    assert(last->type == END_FUNCTION);
    auto epilogue = body.insert(last);
    epilogue->make_skip();
    epilogue->location = last->location;
    epilogue->function = last->function;
    return epilogue;
  }

  /// For each region, a dispatch block placed just after its skip-handlers
  /// GOTO (so normal completion bypasses it): branch to the first matching
  /// handler, else propagate to the enclosing region / epilogue.
  void build_dispatch(
    goto_programt &body,
    std::vector<regiont> &regions,
    goto_programt::targett epilogue)
  {
    for (regiont &r : regions)
    {
      r.dispatch = body.insert(std::next(std::next(r.pop)));
      r.dispatch->make_skip();
      r.dispatch->location = r.pop->location;
      r.dispatch->function = r.pop->function;
    }

    for (regiont &r : regions)
    {
      auto pos = std::next(r.dispatch);
      auto add = [&]() {
        auto n = body.insert(pos);
        n->location = r.dispatch->location;
        n->function = r.dispatch->function;
        return n;
      };
      bool catch_all = false;
      for (const handlert &h : r.handlers)
      {
        auto g = add();
        if (h.type == ellipsis_id)
        {
          g->make_goto(h.landing);
          catch_all = true;
          break;
        }
        // A dispatch block is only ever reached with an exception in flight
        // (thrown == true), so the type-match guard alone is sufficient.
        g->make_goto(h.landing, match_guard(h.type));
      }
      if (!catch_all)
        add()->make_goto(target_for(regions, r.parent, epilogue));
    }
  }

  /// Insert `__ESBMC_exc_thrown = false` before a handler and rewrite its
  /// `var = NONDET` binding to read the thrown object via __ESBMC_exc_value:
  ///   catch (T &v): v is a T* — bind the address      v = (T*)value
  ///   catch (T  v): v is a T  — copy (slice) the object v = *(T*)value
  /// (pointer catches `catch (T*)` are filtered out earlier, so a pointer-typed
  /// bound variable is always a reference catch.)
  goto_programt::targett make_landing(goto_programt &body, const handlert &h)
  {
    auto landing = body.insert(h.target);
    landing->make_assignment();
    landing->code = code_assign2tc(thrown, gen_false_expr());
    landing->location = h.target->location;
    landing->function = h.target->function;

    // The handler body begins after the catch marker, plus the parameter binding
    // for a typed catch. The decrement of the uncaught count goes there, so it
    // happens once the exception has entered its handler ([except.uncaught]) and
    // after the catch-parameter is bound (§5.5).
    auto before_body = h.target;
    if (h.type != ellipsis_id)
    {
      auto bind = std::next(h.target);
      const expr2tc &var = to_code_assign2t(bind->code).target;
      // A reference catch `catch (T &)` binds the address: var (a T*) = value.
      // A value catch `catch (T)` and a pointer catch `catch (T*)` both copy
      // the stored object/pointer out: var = *(decltype(var)*)value. The two
      // pointer-typed forms are told apart by the catch type's `_ptr` suffix.
      bool ref_catch = is_pointer_type(var->type) && !is_pointer_catch(h.type);
      expr2tc src =
        ref_catch
          ? typecast2tc(var->type, value)
          : dereference2tc(
              var->type, typecast2tc(pointer_type2tc(var->type), value));
      bind->code = code_assign2tc(var, src);
      before_body = bind;
    }

    auto insert_after = before_body;
    if (track_uncaught_)
    {
      auto dec = body.insert(std::next(insert_after));
      dec->make_assignment();
      dec->code = adjust_uncaught(-1);
      dec->location = h.target->location;
      dec->function = h.target->function;
      insert_after = dec;
    }

    expr2tc push = make_c_helper_call(exception_globals::push_handled_id);
    if (!is_nil_expr(push))
    {
      auto p = body.insert(std::next(insert_after));
      p->make_function_call(push);
      p->location = h.target->location;
      p->function = h.target->function;
    }
    return landing;
  }

  void
  build_handled_stack(goto_programt &body, const std::vector<regiont> &regions)
  {
    expr2tc pop = make_c_helper_call(exception_globals::pop_handled_id);
    if (is_nil_expr(pop))
      return;

    for (const regiont &r : regions)
    {
      auto p = body.insert(r.after_try);
      p->make_function_call(pop);
      p->location = r.after_try->location;
      p->function = r.after_try->function;
    }
  }

  /// `typeid in { id(T) : T <: catch_type }`.
  expr2tc match_guard(const irep_idt &catch_type)
  {
    expr2tc disj;
    for (unsigned id : registry.concrete_subtype_ids(catch_type))
    {
      expr2tc eq =
        equality2tc(type_id, constant_int2tc(type_id->type, BigInt(id)));
      disj = is_nil_expr(disj) ? eq : or2tc(disj, eq);
    }
    return is_nil_expr(disj) ? gen_false_expr() : disj;
  }

  /// `__ESBMC_exc_typeid ∈ { id(T) : T <: some allowed type }` — the in-flight
  /// exception is permitted by a dynamic exception specification. Nil when the
  /// specification allows nothing (a no-throw spec), so callers fall back to the
  /// plain `!thrown` boundary check.
  expr2tc spec_guard(const std::vector<irep_idt> &spec_types)
  {
    expr2tc disj;
    for (const irep_idt &t : spec_types)
      for (unsigned id : registry.concrete_subtype_ids(t))
      {
        expr2tc eq =
          equality2tc(type_id, constant_int2tc(type_id->type, BigInt(id)));
        disj = is_nil_expr(disj) ? eq : or2tc(disj, eq);
      }
    return disj;
  }

  /// A call to the installed std::unexpected handler — the set_unexpected
  /// builtin records it as __ESBMC_unexpected — or nil when none is installed.
  expr2tc make_unexpected_call()
  {
    const symbolt *h = ns.lookup("c:@F@__ESBMC_unexpected");
    if (!h)
      return expr2tc();
    code_function_callt fc;
    fc.function() = h->get_value();
    expr2tc call;
    migrate_expr(fc, call);
    return call;
  }

  /// A call to the std::terminate() operational model, or nil when it is not
  /// linked into the program. The OM loads current_terminate_handler (honouring
  /// std::set_terminate), calls it, and asserts on return/throw; its default
  /// handler asserts "terminate called after throwing an exception". The OM is
  /// only present when the program references the exception library — which it
  /// must to install a custom handler — so its absence means the default
  /// (assert) behaviour, which emit_terminate falls back to.
  /// Emit a lowering-synthesized terminate point just before @p before. These
  /// (noexcept/throw-spec violation, uncaught exception at the program entry,
  /// bare throw with no active exception) are verification errors in ESBMC's
  /// model, so they are asserted directly — NOT routed through the OM
  /// std::terminate() — and reported as FAILED. Routing through the OM would let
  /// a custom std::set_terminate handler that ends the path (e.g. abort(),
  /// modeled as assume(0)) silently swallow the violation, a false negative. A
  /// user-written std::terminate() call is an ordinary function call into the OM
  /// and is unaffected. @p skip_cond is the condition under which execution
  /// continues past the point without terminating (nil = always terminate); the
  /// assertion is assert(skip_cond), or assert(false) when skip_cond is nil. @p
  /// reason selects the diagnostic comment. Returns the inserted instruction so
  /// callers can target it with a goto.
  goto_programt::targett emit_terminate(
    goto_programt &body,
    goto_programt::targett before,
    const expr2tc &skip_cond,
    const locationt &loc,
    const irep_idt &fn,
    exception_globals::terminate_reasont reason)
  {
    auto setmeta = [&](goto_programt::targett n) {
      n->location = loc;
      n->function = fn;
    };

    // A lowering-synthesized terminate point (a noexcept/throw-spec violation or
    // an uncaught exception at the program entry) is a verification error in
    // ESBMC's model regardless of any installed std::set_terminate handler
    // ([except.terminate]: reaching std::terminate is abnormal termination, and
    // a handler that returns is itself undefined). Assert the violation directly
    // so it is reported as FAILED — routing it through the OM std::terminate()
    // would let a custom handler that ends the path (e.g. abort(), modeled as
    // assume(0)) silently swallow the violation, a false negative. A *user*
    // `std::terminate()` call is an ordinary function call into the OM and is
    // unaffected by this path; only the dynamic-spec/noexcept/uncaught checks
    // synthesized here use emit_terminate.
    auto first = body.insert(before);
    first->make_assertion(
      is_nil_expr(skip_cond) ? gen_false_expr() : skip_cond);
    setmeta(first);
    first->location.property("exception");
    switch (reason)
    {
    case exception_globals::terminate_reason_uncaught:
      first->location.comment("uncaught exception");
      break;
    case exception_globals::terminate_reason_noexcept:
      first->location.comment("noexcept specification violated");
      break;
    case exception_globals::terminate_reason_exception_spec:
      first->location.comment("exception specification violated");
      break;
    case exception_globals::terminate_reason_no_active:
      first->location.comment("throw with no active exception");
      break;
    default:
      first->location.comment("terminate called after throwing an exception");
      break;
    }
    return first;
  }

  /// Enforce a dynamic exception specification throw(allowed...) (including the
  /// empty throw()) at the epilogue. When an exception the spec does not permit
  /// is propagating out, run the std::unexpected handler and re-check: a handler
  /// that rethrows a permitted type lets it propagate; anything else (handler
  /// returns, rethrows a disallowed type, or no handler installed) is a
  /// violation. Mirrors the imperative goto_symext path: one handler call, no
  /// std::bad_exception substitution.
  void build_dynamic_spec_check(
    goto_programt &body,
    goto_programt::targett epilogue,
    const std::vector<irep_idt> &allowed,
    bool is_entry)
  {
    const locationt loc = epilogue->location;
    const irep_idt fn = epilogue->function;
    auto last = std::next(epilogue); // END_FUNCTION

    auto ins = [&](goto_programt::targett pos) {
      auto n = body.insert(pos);
      n->location = loc;
      n->function = fn;
      return n;
    };
    auto not_thrown = [&]() { return equality2tc(thrown, gen_false_expr()); };
    // spec_guard returns nil for the empty spec throw() (nothing permitted).
    auto permitted = [&]() {
      expr2tc g = spec_guard(allowed);
      return is_nil_expr(g) ? gen_false_expr() : g;
    };

    // Where a permitted (or absent) exception continues. For an entry function
    // any escape is uncaught, so route there to an uncaught assertion instead
    // of returning it to a (non-existent) caller.
    goto_programt::targett ret = last;
    if (is_entry)
      ret = emit_terminate(
        body,
        last,
        not_thrown(),
        loc,
        fn,
        exception_globals::terminate_reason_uncaught);

    // The violation point: always terminate (reached only on a genuine
    // disallowed escape, including the handler returning without throwing).
    auto fail = emit_terminate(
      body,
      ret,
      expr2tc(),
      loc,
      fn,
      exception_globals::terminate_reason_exception_spec);

    // No exception in flight -> done.
    ins(fail)->make_goto(ret, not_thrown());
    // Permitted by the specification -> let it propagate.
    ins(fail)->make_goto(ret, permitted());

    // Disallowed: run the unexpected handler (if any) and re-check.
    expr2tc call = make_unexpected_call();
    if (!is_nil_expr(call))
    {
      // Clear `thrown` so the handler runs with no exception in flight; keep
      // typeid/value so a bare `throw;` in the handler rethrows the original.
      auto clear = ins(fail);
      clear->make_assignment();
      clear->code = code_assign2tc(thrown, gen_false_expr());

      ins(fail)->make_function_call(call);

      // Handler returned without throwing -> the specification is violated.
      ins(fail)->make_goto(fail, not_thrown());

      // Past the guard above the handler must have thrown, so wire_throw has
      // already counted its throw as a fresh uncaught exception. That throw
      // *replaces* the original in-flight exception ([except.unexpected]), so
      // drop the original's contribution to the per-thread uncaught count —
      // exactly one exception is uncaught after the replacement, not two.
      if (track_uncaught_)
      {
        auto a_cnt = ins(fail);
        a_cnt->make_assignment();
        a_cnt->code = adjust_uncaught(-1);
      }

      // Handler rethrew a permitted type -> let it propagate.
      ins(fail)->make_goto(ret, permitted());
    }
    // Fall through to fail.
  }

  /// Replace a throw with: arm the globals (or, for a rethrow, just re-raise
  /// the in-flight one) and branch to the enclosing dispatch / epilogue.
  void wire_throw(
    goto_programt &body,
    goto_programt::targett thr,
    goto_programt::targett dest)
  {
    const code_cpp_throw2t throw_ref = to_code_cpp_throw2t(thr->code); // copy
    const locationt loc = thr->location;
    const irep_idt fn = thr->function;

    auto pos = std::next(thr);
    auto add = [&]() {
      auto n = body.insert(pos);
      n->location = loc;
      n->function = fn;
      return n;
    };

    // A bare `throw;` re-raises the exception currently being handled. When the
    // handled-stack OM is linked (C/C++), the helper re-raises from that stack
    // and calls std::terminate if none is being handled ([except.throw]/9).
    if (is_nil_expr(throw_ref.operand))
    {
      expr2tc call = make_c_helper_call(exception_globals::rethrow_current_id);
      if (!is_nil_expr(call))
      {
        thr->make_function_call(call);
        add()->make_goto(dest);
        return;
      }
      // Fallback without the handled-stack OM (e.g. Python): re-raise inline
      // from the globals — typeid/value still hold the active exception
      // (clear-on-catch reset only `thrown`); with none in flight, terminate.
      thr->make_skip();
      auto reraise = add();
      reraise->make_assignment();
      reraise->code = code_assign2tc(thrown, gen_true_expr());
      if (track_uncaught_)
      {
        auto a_cnt = add();
        a_cnt->make_assignment();
        a_cnt->code = adjust_uncaught(+1);
      }
      add()->make_goto(dest);
      expr2tc has_exc =
        notequal2tc(type_id, constant_int2tc(type_id->type, BigInt(0)));
      emit_terminate(
        body,
        reraise,
        has_exc,
        loc,
        fn,
        exception_globals::terminate_reason_no_active);
      return;
    }

    thr->make_assignment();
    thr->code = code_assign2tc(thrown, gen_true_expr());

    // A real throw arms the globals from the thrown object.
    {
      const unsigned tid = registry.id_of(throw_ref.exception_list.front());
      auto a_tid = add();
      a_tid->make_assignment();
      a_tid->code =
        code_assign2tc(type_id, constant_int2tc(type_id->type, BigInt(tid)));

      // Copy the thrown object into a stable static slot, then point the global
      // at the copy — the operand is a temporary whose frame may be gone by the
      // time a handler (possibly in a caller) reads it.
      expr2tc storage = make_exception_storage(throw_ref.operand->type);
      auto a_copy = add();
      a_copy->make_assignment();
      a_copy->code = code_assign2tc(storage, throw_ref.operand);

      auto a_val = add();
      a_val->make_assignment();
      expr2tc addr = address_of2tc(storage->type, storage);
      a_val->code = code_assign2tc(value, typecast2tc(value->type, addr));
    }

    // The new exception is now uncaught until it reaches its handler.
    if (track_uncaught_)
    {
      auto a_cnt = add();
      a_cnt->make_assignment();
      a_cnt->code = adjust_uncaught(+1);
    }

    add()->make_goto(dest);
  }

  /// After a call that may throw, branch to the enclosing dispatch / epilogue
  /// if the callee left an exception in flight.
  void wire_call(
    goto_programt &body,
    goto_programt::targett call,
    goto_programt::targett dest)
  {
    auto g = body.insert(std::next(call));
    g->make_goto(dest, thrown);
    g->location = call->location;
    g->function = call->function;
  }
};
} // namespace

void remove_exceptions(
  goto_functionst &goto_functions,
  contextt &context,
  const namespacet &ns)
{
  // A program with no throw/catch needs no exception machinery. Skip entirely so
  // the pass is a true no-op for exception-free programs — otherwise it would
  // add the exception-state globals to __ESBMC_main, perturbing analyses that
  // inspect program state on programs that have nothing to do with exceptions
  // (e.g. termination's recurrent-set search, function-contract frames).
  if (!exception_loweringt::program_uses_exceptions(goto_functions))
    return;
  create_exception_state_symbols(context);
  exception_loweringt(context, ns).run(goto_functions);
}
