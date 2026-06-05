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
#include <irep2/irep2_utils.h>

namespace
{
const irep_idt ellipsis_id = "ellipsis";
const irep_idt noexcept_id = "noexcept";

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
    // Turn dynamic_cast<T&> bad_cast intrinsics into ordinary THROWs first, so
    // every later step (may-throw, registry, dispatch) treats them uniformly.
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
      for (const auto &ins : fn.second.body.instructions)
      {
        if (ins.type == THROW)
        {
          const code_cpp_throw2t &t = to_code_cpp_throw2t(ins.code);
          if (!is_nil_expr(t.operand))
            registry.register_chain(t.exception_list);
        }
        else if (ins.type == THROW_DECL)
        {
          // A dynamic exception specification's allowed types must each have an
          // id so the epilogue can test the in-flight exception's membership.
          for (const irep_idt &ty :
               to_code_cpp_throw_decl2t(ins.code).exception_list)
            if (ty != noexcept_id)
              registry.register_chain({ty});
        }
        else if (ins.type == FUNCTION_CALL)
        {
          const code_function_call2t &c = to_code_function_call2t(ins.code);
          if (
            is_symbol2t(c.function) &&
            id2string(to_symbol2t(c.function).thename).find("pthread_create") !=
              std::string::npos)
            return; // concurrent program — not yet modelled
        }
      }
    }

    // Whole-program, all-or-nothing: lowered and imperative dispatch cannot
    // interoperate across a call, so unless every participating function is in
    // the supported subset we leave the entire program to symex.
    for (auto &fn : goto_functions.function_map)
      if (fn.second.body_available && !program_supported(fn.second.body))
        return;

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
      return;

    for (auto &fn : goto_functions.function_map)
      if (fn.second.body_available)
        lower_ip(fn.second.body, fn.first);

    // The globals are created after the frontend builds __ESBMC_main's static
    // initialisation, so seed them at entry; otherwise symex does not track the
    // cross-frame writes that carry an exception out of a callee.
    auto entry = goto_functions.function_map.find("__ESBMC_main");
    if (
      entry != goto_functions.function_map.end() &&
      entry->second.body_available)
      init_globals(entry->second.body);
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

  // ---- bad_cast lowering -------------------------------------------------

  /// THROW code for a synthesized std::bad_cast (operand + exception_list =
  /// dynamic type and its bases), or nil if the <typeinfo> model's class is not
  /// in the symbol table. Mirrors goto_symext::symex_throw_bad_cast; the tag is
  /// elaborated ("class std::bad_cast") on newer Clang/LLVM, un-elaborated on
  /// older, so try both. Built once and reused across call sites (a single
  /// exception is in flight at a time).
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
  /// failing dynamic_cast<T&> lowers to — with the equivalent THROW, so the rest
  /// of the pass lowers it like any other throw. If std::bad_cast is
  /// unresolvable the calls are left in place and program_supported's bad_cast
  /// guard makes the program fall back to the imperative path.
  void lower_bad_cast_calls(goto_functionst &goto_functions)
  {
    expr2tc bad_cast_throw = build_bad_cast_throw();
    if (is_nil_expr(bad_cast_throw))
      return;

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
          is_symbol2t(c.function) &&
          to_symbol2t(c.function).thename == "c:@F@__ESBMC_throw_bad_cast")
        {
          // make_throw() preserves location/function (clear() leaves them).
          ins.make_throw();
          ins.code = bad_cast_throw;
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
  bool program_supported(goto_programt &body)
  {
    const auto end = body.instructions.end();
    int depth = 0;
    for (auto it = body.instructions.begin(); it != end; ++it)
    {
      // Both no-throw specs (noexcept / throw()) and dynamic exception
      // specifications (throw(T...)) are enforced at the function epilogue
      // (see lower_ip), so a THROW_DECL never forces fallback.
      if (it->type == CATCH && !it->targets.empty())
      {
        const code_cpp_catch2t &c = to_code_cpp_catch2t(it->code);
        if (c.exception_list.size() != it->targets.size())
          return false;
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
            return false;
          auto bind = std::next(tgt);
          if (bind == end || !is_code_assign2t(bind->code))
            return false;
        }
        ++depth;
      }
      else if (it->type == CATCH)
      {
        // pop: needs the skip-handlers GOTO after it for dispatch placement.
        auto nx = std::next(it);
        if (depth == 0 || nx == end || nx->type != GOTO)
          return false;
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
          return false;
      }
      else if (it->type == FUNCTION_CALL)
      {
        // __ESBMC_throw_bad_cast is a bodyless intrinsic that symex turns into a
        // std::bad_cast throw using the imperative stack_catch — which is empty
        // once the surrounding catch is lowered, so the throw would be reported
        // uncaught. The lowering cannot see this hidden throw to wire it, so
        // leave such a program (dynamic_cast<T&>) to the imperative path.
        const code_function_call2t &c = to_code_function_call2t(it->code);
        if (
          is_symbol2t(c.function) &&
          to_symbol2t(c.function).thename == "c:@F@__ESBMC_throw_bad_cast")
          return false;
      }
    }
    return depth == 0;
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

  void lower_ip(goto_programt &body, const irep_idt &fn_id)
  {
    std::vector<regiont> regions;
    std::vector<sitet> throws, calls;
    collect(body, regions, throws, calls);

    // Throw-spec markers are consumed here; once the throws are lowered the
    // imperative throw-decl machinery has nothing to act on. Record the
    // function's exception specification (if any) in the same pass: `has_spec`
    // is set by any THROW_DECL, and `spec_types` collects its allowed types
    // (empty for a no-throw spec — noexcept / throw()).
    bool has_spec = false;
    std::vector<irep_idt> spec_types;
    for (auto &ins : body.instructions)
    {
      if (ins.type == THROW_DECL)
      {
        has_spec = true;
        for (const irep_idt &ty :
             to_code_cpp_throw_decl2t(ins.code).exception_list)
          if (ty != noexcept_id)
            spec_types.push_back(ty);
        ins.make_skip();
      }
      else if (ins.type == THROW_DECL_END)
        ins.make_skip();
    }

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

    // An exception in flight at the epilogue is a hard failure in two cases:
    //  - main / __ESBMC_main: it is uncaught (escapes the program → terminate);
    //    __ESBMC_main also covers static-init throws from global constructors.
    //  - an exception specification is violated: the escaping exception's type
    //    is not permitted by the function's `throw(...)` / noexcept declaration
    //    ([except.spec]). A no-throw spec permits nothing, so any escape fails;
    //    a dynamic spec permits its listed types (and their subtypes). This
    //    mirrors the imperative path, which reports an escaping out-of-spec
    //    throw as "not allowed by declaration" (unexpected()/handler dispatch is
    //    not modelled on either path). A locally-caught throw clears `thrown`,
    //    so the check fires only on a genuine escape.
    const bool is_entry =
      id2string(fn_id).rfind("c:@F@main#", 0) == 0 || fn_id == "__ESBMC_main";
    if (is_entry || has_spec)
    {
      // The assertion is `!thrown || in_spec`; for an uncaught (entry) or
      // no-throw boundary `in_spec` is empty, leaving the original `!thrown`.
      expr2tc not_thrown = equality2tc(thrown, gen_false_expr());
      expr2tc in_spec = is_entry ? expr2tc() : spec_guard(spec_types);
      auto a = body.insert(std::next(epilogue));
      a->make_assertion(
        is_nil_expr(in_spec) ? not_thrown : or2tc(not_thrown, in_spec));
      a->location = epilogue->location;
      a->function = epilogue->function;
      a->location.property("exception");
      // A no-throw spec (empty allowed set) keeps the distinct "noexcept"
      // wording; a dynamic spec reports a specification violation.
      a->location.comment(
        is_entry             ? "uncaught exception"
        : spec_types.empty() ? "noexcept specification violated"
                             : "exception specification violated");
    }

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
