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
/// globals (issue #5075). P1 scope: per-function, all-or-nothing, intra-
/// procedural, reference catches, dtor-free, with correct nested-try
/// propagation (a throw is matched against its enclosing try chain from the
/// innermost outward). Functions outside the subset are left for symex.
class exception_loweringt
{
public:
  exception_loweringt(contextt &context, const namespacet &ns)
    : context(context),
      ns(ns),
      registry(ns),
      thrown(mk_global(exception_globals::thrown_id)),
      type_id(mk_global(exception_globals::typeid_id)),
      value(mk_global(exception_globals::value_id))
  {
  }

  void run(goto_functionst &goto_functions)
  {
    // Rewrite dynamic_cast<T&> bad_cast intrinsics into real THROWs first (the
    // std::bad_cast type is synthesized if <typeinfo> was not included), so every
    // later step (may-throw, registry, dispatch, program_supported) treats them
    // uniformly and the pass never falls back on a bad_cast site.
    lower_bad_cast_calls(goto_functions);

    may_throw = compute_may_throw(goto_functions);

    // Single scan: teach the registry any exception hierarchy that lives only in
    // THROW exception_lists (the Python frontend's classes have no `tag-`
    // symbol), and detect concurrency. The exception state is one global tuple,
    // not per-thread, so the lowered dispatch is unsound for concurrent programs
    // (one thread could observe, catch, or clear another thread's in-flight
    // exception). Until the state is modeled per thread, leave concurrent
    // programs to the imperative path.
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
          if (
            is_symbol2t(c.function) &&
            id2string(to_symbol2t(c.function).thename).find("pthread_create") !=
              std::string::npos)
            // concurrent program — not yet modelled
            return report_fallback(
              goto_functions, "a concurrent program (pthread_create)");
        }
      }
    }

    // Whole-program, all-or-nothing: lowered and imperative dispatch cannot
    // interoperate across a call, so unless every participating function is in
    // the supported subset we leave the entire program to symex.
    for (auto &fn : goto_functions.function_map)
      if (fn.second.body_available && !program_supported(fn.second.body))
        return report_fallback(goto_functions, unsupported_reason_, fn.first);

    // The uncaught-exception check is anchored at the program entry's epilogue.
    // __ESBMC_main is the universal whole-program entry (it runs static init,
    // then calls main / python_user_main), so an escaping exception always
    // reaches it. Without it (e.g. --function isolated verification) an uncaught
    // exception could be silently accepted, so fall back to the imperative path.
    bool has_entry = false;
    for (const auto &fn : goto_functions.function_map)
      if (fn.first == "__ESBMC_main")
        has_entry = true;
    if (!has_entry)
      return report_fallback(
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

  /// True if any function still has an exception construct to lower.
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

  /// Report that the pass declined to lower a program that uses exceptions, so
  /// the residual dependence on the imperative path is visible rather than
  /// silent. Today the imperative path takes over; once it is removed (P4) this
  /// diagnostic is what keeps that deletion non-breaking — an unsupported
  /// program is reported, not miscompiled. Programs with no exception construct
  /// are silent (the pass is a no-op for them).
  void report_fallback(
    const goto_functionst &gf,
    const std::string &why,
    const irep_idt &fn = irep_idt())
  {
    if (!program_uses_exceptions(gf))
      return;
    if (fn.empty())
      log_warning(
        "--lower-exceptions: cannot lower {}; falling back to the imperative "
        "exception path",
        why);
    else
      log_warning(
        "--lower-exceptions: cannot lower {} in '{}'; falling back to the "
        "imperative exception path",
        why,
        id2string(fn));
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
    body.update();
  }

private:
  contextt &context;
  const namespacet &ns;
  exception_typeidt registry;
  expr2tc thrown, type_id, value;
  std::set<irep_idt> may_throw;
  unsigned storage_counter = 0;

  expr2tc mk_global(const char *id)
  {
    const symbolt *s = ns.lookup(irep_idt(id));
    assert(s && "exception-state globals must be created before lowering");
    return symbol2tc(migrate_type(s->get_type()), s->id);
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
    context.move_symbol_to_context(sym);
    return symbol2tc(obj_type, irep_idt(id));
  }

  // ---- runtime-thrown std exceptions (bad_cast, bad_exception) -----------

  /// Look up a std exception class by its two possible tag spellings (elaborated
  /// "class std::X" on newer Clang/LLVM, un-elaborated "std::X" on older), or
  /// synthesize a minimal empty struct when absent. A compiled program throws
  /// these runtime exceptions (std::bad_cast on a failing dynamic_cast<T&>,
  /// std::bad_exception on a violated dynamic spec) whether or not <typeinfo> /
  /// <exception> was included — the runtime constructs the object; header
  /// visibility only governs whether the type can be *named* (catch(std::X&) /
  /// typeid). With a synthesized type a catch(...) still catches it and an
  /// uncaught one still terminates, matching real semantics. @p name is the
  /// un-elaborated class name, e.g. "std::bad_cast".
  const symbolt *get_exception_class(const std::string &name)
  {
    const symbolt *sym = ns.lookup("tag-" + name);
    if (!sym)
      sym = ns.lookup("tag-class " + name);
    if (sym)
      return sym;

    symbolt s;
    s.id = "tag-" + name;
    s.name = name;
    s.mode = "C++";
    struct_typet st;
    st.set("tag", name);
    s.set_type(st);
    s.is_type = true;
    return context.move_symbol_to_context(s);
  }

  /// THROW code for the given exception class (operand in a stable static slot +
  /// exception_list = the type and its bases). Built once per type and reused (a
  /// single exception is in flight at a time).
  expr2tc build_exception_throw(const symbolt *sym)
  {
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

    // A static slot holds the thrown object (its state is irrelevant — only its
    // type identity and address matter to the handler).
    expr2tc operand = make_exception_storage(migrate_symbol_type(*sym));
    return code_cpp_throw2tc(operand, exception_list);
  }

  expr2tc build_bad_cast_throw()
  {
    return build_exception_throw(get_exception_class("std::bad_cast"));
  }

  /// Rewrite each __ESBMC_throw_bad_cast() call — the bodyless intrinsic a
  /// failing dynamic_cast<T&> lowers to — into a real THROW of a materialized
  /// std::bad_cast, so the rest of the pass lowers it like any other throw and
  /// never falls back for it. The throw is built lazily on the first site (so a
  /// program with no dynamic_cast<T&> never synthesizes the type) and reused.
  void lower_bad_cast_calls(goto_functionst &goto_functions)
  {
    expr2tc bad_cast_throw;

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

        if (is_nil_expr(bad_cast_throw))
          bad_cast_throw = build_bad_cast_throw();
        // make_throw() preserves location/function (clear() leaves them).
        ins.make_throw();
        ins.code = bad_cast_throw;
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
            callees[fn.first].insert(to_symbol2t(c.function).thename);
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
  /// of registered objects or rethrows, regions whose pop is followed by the
  /// skip-handlers GOTO)? Read-only.
  /// Set when program_supported declines, naming the construct for the
  /// fallback diagnostic (report_fallback).
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
        // pop. A try whose handler bodies are non-empty has a skip-handlers
        // GOTO right after the pop; an empty trailing handler (catch(...){}
        // whose body coincides with the after-try point) has none. Both are
        // supported: normalize_empty_handlers synthesizes the missing skip
        // GOTO before lowering, so build_dispatch always finds it.
        if (depth == 0)
          return unsupported("an unmatched catch pop");
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

  /// An empty trailing handler (`catch(...){}` whose body is empty) compiles to
  /// a try whose CATCH pop is *not* followed by the usual skip-handlers GOTO:
  /// the empty handler's entry coincides with the after-try point, so normal
  /// completion just falls through the pop into it and there is nothing to skip.
  /// build_dispatch, however, places each region's dispatch block right after
  /// that skip GOTO (so normal completion bypasses the dispatch and landing).
  /// Synthesize the missing GOTO after each bare pop, targeting the current
  /// fall-through (= the handler/after-try point), restoring the shape the rest
  /// of the pass expects. Run after rebalance_removed_pops, whose synthetic pops
  /// already carry a skip GOTO, so the only bare pops left are empty handlers.
  void normalize_empty_handlers(goto_programt &body)
  {
    const auto end = body.instructions.end();
    for (auto it = body.instructions.begin(); it != end; ++it)
    {
      if (it->type != CATCH || !it->targets.empty())
        continue; // not a pop
      auto nx = std::next(it);
      if (nx == end || nx->type == GOTO)
        continue; // already has a skip-handlers GOTO
      // Insert a GOTO to the original fall-through; the iterator nx is stable,
      // so it still names the after-try point after later make_landing inserts.
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
    normalize_empty_handlers(body);

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
    //  - noexcept: an escape calls std::terminate ([except.spec]).
    //  - throw(T...) / throw(): a disallowed escape runs std::unexpected and is
    //    re-checked (build_dynamic_spec_check), matching the imperative path
    //    (which models the unexpected-handler dispatch and its recovery).
    // A locally-caught throw clears `thrown`, so these fire only on a genuine
    // escape (terminate iff an exception is still in flight).
    const bool is_entry =
      id2string(fn_id).rfind("c:@F@main#", 0) == 0 || fn_id == "__ESBMC_main";

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
        is_entry ? "uncaught exception" : "noexcept specification violated");

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
    }
    return landing;
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
  expr2tc make_terminate_call()
  {
    const symbolt *h = ns.lookup("c:@N@std@F@terminate#");
    if (!h)
      return expr2tc();
    code_function_callt fc;
    fc.function() = symbol_exprt(h->id, h->get_type());
    expr2tc call;
    migrate_expr(fc, call);
    return call;
  }

  /// Emit a terminate point just before @p before: route through the OM
  /// std::terminate() when it is linked (so a custom std::set_terminate handler
  /// runs), else fall back to an assertion (the established "terminate = property
  /// violation" classification, §5.4). @p skip_cond is the condition under which
  /// execution continues past the point without terminating (nil = always
  /// terminate); the assertion fallback is assert(skip_cond), or assert(false)
  /// when skip_cond is nil. Returns the first inserted instruction so callers can
  /// target it with a goto. std::terminate() never returns (the OM ends every
  /// path in __ESBMC_assume(false)).
  goto_programt::targett emit_terminate(
    goto_programt &body,
    goto_programt::targett before,
    const expr2tc &skip_cond,
    const locationt &loc,
    const irep_idt &fn,
    const char *comment)
  {
    auto setmeta = [&](goto_programt::targett n) {
      n->location = loc;
      n->function = fn;
    };
    expr2tc call = make_terminate_call();
    if (!is_nil_expr(call))
    {
      goto_programt::targett first;
      if (!is_nil_expr(skip_cond))
      {
        first = body.insert(before);
        first->make_goto(before, skip_cond); // no exception in flight -> skip
        setmeta(first);
      }
      // Clear the in-flight exception before terminating: the std::terminate()
      // OM has its own try/catch over the handler call, which keys on the same
      // global, so a leftover `thrown` would corrupt its control flow. We have
      // decided to terminate, so the exception is no longer propagating.
      auto clear = body.insert(before);
      clear->make_assignment();
      clear->code = code_assign2tc(thrown, gen_false_expr());
      setmeta(clear);
      auto c = body.insert(before);
      c->make_function_call(call);
      setmeta(c);
      return is_nil_expr(skip_cond) ? clear : first;
    }

    auto a = body.insert(before);
    a->make_assertion(is_nil_expr(skip_cond) ? gen_false_expr() : skip_cond);
    setmeta(a);
    a->location.property("exception");
    a->location.comment(comment);
    return a;
  }

  /// Enforce a dynamic exception specification throw(allowed...) (including the
  /// empty throw()) at the epilogue. When an exception the spec does not permit
  /// is propagating out, run the std::unexpected handler and re-check: a handler
  /// that rethrows a permitted type lets it propagate. If the handler rethrows a
  /// disallowed type, the in-flight exception is replaced by a std::bad_exception
  /// when the spec lists it (and then propagates, since the spec permits it);
  /// otherwise — disallowed with no std::bad_exception in the spec, the handler
  /// returning, or no handler installed — it is a violation (std::terminate).
  /// See [except.unexpected].
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
    // Does the spec list std::bad_exception? If so a violated spec substitutes
    // it rather than terminating (see below). The name carries the tag spelling
    // convert_exception_id produced, elaborated or not, so accept both.
    const bool allows_bad_exception =
      std::find(allowed.begin(), allowed.end(), "std::bad_exception") !=
        allowed.end() ||
      std::find(allowed.begin(), allowed.end(), "class std::bad_exception") !=
        allowed.end();

    // Where a permitted (or absent) exception continues. For an entry function
    // any escape is uncaught, so terminate there (iff still in flight) instead
    // of returning it to a (non-existent) caller.
    goto_programt::targett ret = last;
    if (is_entry)
      ret =
        emit_terminate(body, last, not_thrown(), loc, fn, "uncaught exception");

    // The violation point: always terminate (reached only on a genuine
    // disallowed escape, including the handler returning without throwing).
    auto fail = emit_terminate(
      body, ret, expr2tc(), loc, fn, "exception specification violated");

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
      // Handler rethrew a permitted type -> let it propagate.
      ins(fail)->make_goto(ret, permitted());

      // Handler rethrew a disallowed type. If the spec lists std::bad_exception
      // the in-flight exception is replaced by a std::bad_exception, which the
      // spec permits, so it propagates ([except.unexpected]); otherwise the
      // fall-through to `fail` models std::terminate. (Reached only with an
      // exception in flight, so `thrown` is already true.)
      if (allows_bad_exception)
      {
        const symbolt *be = get_exception_class("std::bad_exception");
        const code_cpp_throw2t bx =
          to_code_cpp_throw2t(build_exception_throw(be));
        const unsigned tid = registry.id_of(bx.exception_list.front());
        auto a_tid = ins(fail);
        a_tid->make_assignment();
        a_tid->code =
          code_assign2tc(type_id, constant_int2tc(type_id->type, BigInt(tid)));
        auto a_val = ins(fail);
        a_val->make_assignment();
        expr2tc addr = address_of2tc(bx.operand->type, bx.operand);
        a_val->code = code_assign2tc(value, typecast2tc(value->type, addr));
        ins(fail)->make_goto(ret);
      }
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

    thr->make_assignment();
    thr->code = code_assign2tc(thrown, gen_true_expr());

    auto pos = std::next(thr);
    auto add = [&]() {
      auto n = body.insert(pos);
      n->location = loc;
      n->function = fn;
      return n;
    };

    // A rethrow (`throw;`) re-raises the current exception: the typeid/value
    // globals already hold it (clear-on-catch only reset `thrown`).
    if (!is_nil_expr(throw_ref.operand))
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
  create_exception_state_symbols(context);
  exception_loweringt(context, ns).run(goto_functions);
}
