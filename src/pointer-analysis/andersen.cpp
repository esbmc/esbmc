#include <pointer-analysis/andersen.h>

#include <irep2/irep2_utils.h>
#include <util/lang/c_types.h>

namespace
{
expr2tc strip_casts(const expr2tc &e)
{
  expr2tc cur = e;
  while (is_typecast2t(cur) || is_bitcast2t(cur))
    cur = is_typecast2t(cur) ? to_typecast2t(cur).from : to_bitcast2t(cur).from;
  return cur;
}

/// Reduces an access to the object it reaches into.  The analysis is field-
/// and index-insensitive, so every member, element and byte-level selection
/// folds onto the base object's node.
expr2tc base_object(const expr2tc &e)
{
  expr2tc cur = e;
  while (true)
  {
    if (is_member2t(cur))
      cur = to_member2t(cur).source_value;
    else if (is_index2t(cur))
      cur = to_index2t(cur).source_value;
    else if (is_byte_extract2t(cur))
      cur = to_byte_extract2t(cur).source_value;
    else if (is_typecast2t(cur))
      cur = to_typecast2t(cur).from;
    else if (is_bitcast2t(cur))
      cur = to_bitcast2t(cur).from;
    else
      return cur;
  }
}

/// Whether a value of this type can hold an address.  Field insensitivity
/// means a whole-struct copy still moves the pointers buried inside it.
bool may_carry_pointer(const type2tc &t)
{
  if (is_nil_type(t))
    return false;
  if (is_pointer_type(t))
    return true;
  // An unresolved symbolic type could be anything; skipping the assignment
  // would under-approximate, so assume the worst and let it widen to TOP.
  if (is_symbol_type(t))
    return true;
  if (is_array_type(t))
    return may_carry_pointer(to_array_type(t).subtype);

  if (is_struct_type(t) || is_union_type(t))
  {
    const std::vector<type2tc> &members =
      is_struct_type(t) ? to_struct_type(t).members : to_union_type(t).members;
    for (const type2tc &m : members)
      if (may_carry_pointer(m))
        return true;
  }
  return false;
}

expr2tc top_source_expr()
{
  return symbol2tc(pointer_type2(), "andersen::top_source");
}

expr2tc return_symbol(const irep_idt &fn)
{
  return symbol2tc(pointer_type2(), "andersen::return::" + fn.as_string());
}
} // namespace

andersent::andersent()
{
  // Reserve node 0 as TOP ("may point anywhere").  It owns a slot in every
  // node-indexed vector but is never handed out by get_node, so real program
  // nodes are numbered from 1.  See the soundness contract in andersen.h.
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

andersent::node_id andersent::fresh_node()
{
  return get_node(
    symbol2tc(pointer_type2(), "andersen::tmp" + std::to_string(tmp_count++)));
}

bool andersent::join(node_id dst, node_id src)
{
  if (dst == src)
    return false;

  const std::size_t before = pts[dst].size();
  pts[dst].insert(pts[src].begin(), pts[src].end());
  return pts[dst].size() != before;
}

bool andersent::add_copy_edge(node_id from, node_id to)
{
  copy_edges[from].insert(to);
  return join(to, from);
}

void andersent::points_to_top(node_id n)
{
  ensure_node(n);
  pts[n].insert(TOP);
}

const andersent::node_id *andersent::find_node(const expr2tc &e) const
{
  if (is_symbol2t(e))
  {
    auto it = symbol_to_node.find(to_symbol2t(e).thename);
    return it == symbol_to_node.end() ? nullptr : &it->second;
  }

  auto it = expr_to_node.find(e);
  return it == expr_to_node.end() ? nullptr : &it->second;
}

andersent::node_id andersent::get_node(const expr2tc &e)
{
  if (const node_id *existing = find_node(e))
    return *existing;

  const node_id n = node_to_expr.size();
  // A formal parameter is interned from the callee's signature at the call
  // site and from the body's own symbol inside it; keying on the identifier
  // is what makes those the same node even when the two types differ.
  if (is_symbol2t(e))
    symbol_to_node.emplace(to_symbol2t(e).thename, n);
  else
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
  // Dynamic (load/store) constraints are indexed by the node they dereference,
  // so a work-list pop costs O(uses of that pointer) rather than a scan of
  // every constraint.
  std::vector<std::vector<node_id>> loads(pts.size());  // loads[q]:  p = *q
  std::vector<std::vector<node_id>> stores(pts.size()); // stores[p]: *p = q

  // pts is deliberately not cleared: points_to_top() writes into it directly
  // and is not backed by a constraint.  Re-seeding on top of a previous
  // fixpoint is sound regardless, since sets only ever grow.
  for (auto &edges : copy_edges)
    edges.clear();

  for (const constraintt &c : constraints)
    switch (c.kind)
    {
    case constraint_kindt::ADDRESS_OF:
      pts[c.lhs].insert(c.rhs);
      break;
    case constraint_kindt::COPY:
      copy_edges[c.rhs].insert(c.lhs);
      break;
    case constraint_kindt::LOAD:
      loads[c.rhs].push_back(c.lhs);
      break;
    case constraint_kindt::STORE:
      stores[c.lhs].push_back(c.rhs);
      break;
    }

  std::vector<bool> queued(pts.size(), false);
  std::vector<node_id> worklist;
  const auto enqueue = [&queued, &worklist](node_id n) {
    if (!queued[n])
    {
      queued[n] = true;
      worklist.push_back(n);
    }
  };

  for (node_id n = 0; n < pts.size(); ++n)
    if (!pts[n].empty())
      enqueue(n);

  while (!worklist.empty())
  {
    const node_id n = worklist.back();
    worklist.pop_back();
    queued[n] = false;

    // Snapshot: resolving a dereference through n can grow pts[n] itself.
    const std::unordered_set<node_id> targets = pts[n];
    const bool anywhere = targets.count(TOP) != 0;

    for (node_id p : loads[n]) // p = *n
    {
      // There are no nameable objects "inside" TOP to draw an edge from, so a
      // load through a may-point-anywhere pointer yields an unknown value.
      if (anywhere && pts[p].insert(TOP).second)
        enqueue(p);

      for (node_id o : targets)
        if (o != TOP && add_copy_edge(o, p))
          enqueue(p);
    }

    for (node_id q : stores[n]) // *n = q
    {
      // Symmetrically, a store through such a pointer writes to an unknown
      // object; pts[TOP] is the conservative sink.
      if (anywhere && join(TOP, q))
        enqueue(TOP);

      for (node_id o : targets)
        if (o != TOP && add_copy_edge(q, o))
          enqueue(o);
    }

    for (node_id m : copy_edges[n])
      if (join(m, n))
        enqueue(m);
  }

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

andersent::node_id andersent::top_source()
{
  node_id n = get_node(top_source_expr());
  points_to_top(n);
  return n;
}

andersent::node_id andersent::return_node(const irep_idt &fn)
{
  return get_node(return_symbol(fn));
}

andersent::node_id andersent::eval_rhs(const expr2tc &rhs, unsigned loc)
{
  const expr2tc r = strip_casts(rhs);

  if (is_address_of2t(r))
  {
    const expr2tc &obj = to_address_of2t(r).ptr_obj;
    const node_id t = fresh_node();
    if (is_dereference2t(obj)) // &*q is just q
      add_constraint(
        constraint_kindt::COPY,
        t,
        get_node(base_object(to_dereference2t(obj).value)));
    else
      add_constraint(
        constraint_kindt::ADDRESS_OF, t, get_node(base_object(obj)));
    return t;
  }

  if (is_dereference2t(r))
  {
    const node_id t = fresh_node();
    add_constraint(
      constraint_kindt::LOAD,
      t,
      get_node(base_object(to_dereference2t(r).value)));
    return t;
  }

  if (is_sideeffect2t(r))
  {
    const sideeffect2t &side = to_sideeffect2t(r);
    switch (side.kind)
    {
    case sideeffect2t::allockind::malloc:
    case sideeffect2t::allockind::realloc:
    case sideeffect2t::allockind::alloca:
    case sideeffect2t::allockind::cpp_new:
    case sideeffect2t::allockind::cpp_new_arr:
    {
      // One node per allocation site, so every object a loop allocates shares
      // a single abstraction.  Keyed like value_sett's dynamic objects.
      const type2tc &objtype = is_pointer_type(side.type)
                                 ? to_pointer_type(side.type).subtype
                                 : side.alloctype;
      const node_id t = fresh_node();
      add_constraint(
        constraint_kindt::ADDRESS_OF,
        t,
        get_node(dynamic_object2tc(objtype, gen_ulong(loc), false, false)));
      return t;
    }

    case sideeffect2t::allockind::nondet:
    {
      // A nondet *pointer* is an unconstrained bit pattern that symbolic
      // execution may later constrain to equal an existing object's address,
      // so it may name anything.  Only a value that cannot carry a pointer at
      // all names nothing.
      const node_id t = fresh_node();
      if (may_carry_pointer(side.type))
        points_to_top(t);
      return t;
    }

    default:
      break;
    }
  }

  if (is_if2t(r))
  {
    const if2t &branch = to_if2t(r);
    const node_id t = fresh_node();
    add_constraint(constraint_kindt::COPY, t, eval_rhs(branch.true_value, loc));
    add_constraint(
      constraint_kindt::COPY, t, eval_rhs(branch.false_value, loc));
    return t;
  }

  // Pointer arithmetic keeps pointing into the same object: this analysis is
  // offset-insensitive, so it is a plain copy of the pointer operand.
  if (is_add2t(r) || is_sub2t(r))
  {
    const expr2tc &lhs = *r->get_sub_expr(0);
    const expr2tc &rhs_op = *r->get_sub_expr(1);
    if (is_pointer_type(lhs->type))
      return eval_rhs(lhs, loc);
    if (is_pointer_type(rhs_op->type))
      return eval_rhs(rhs_op, loc);
  }

  // NULL names no object, so an empty set is exact.  A *non-zero* integer
  // turned into a pointer is an int->ptr cast that may name anything, so it
  // deliberately falls through to TOP below.
  if (
    is_null_object2t(r) ||
    (is_constant_int2t(r) && to_constant_int2t(r).value.is_zero()))
    return fresh_node();

  // A nameable l-value read as a value: its own node already holds its targets.
  if (
    is_symbol2t(r) || is_member2t(r) || is_index2t(r) || is_dynamic_object2t(r))
    return get_node(base_object(r));

  // Anything else is a value this frontend does not model.  Leaving the set
  // empty would be an unsound under-approximation; TOP is the safe answer.
  const node_id t = fresh_node();
  points_to_top(t);
  return t;
}

void andersent::assign_top(const expr2tc &lhs)
{
  const expr2tc target = base_object(lhs);

  if (is_dereference2t(target))
    add_constraint(
      constraint_kindt::STORE,
      get_node(base_object(to_dereference2t(target).value)),
      top_source());
  else
    points_to_top(get_node(target));
}

void andersent::handle_assign(
  const expr2tc &lhs,
  const expr2tc &rhs,
  unsigned loc)
{
  if (is_nil_expr(lhs))
    return;

  // A nil source is an unconstrained value, so route the target to TOP rather
  // than dropping the assignment.
  if (is_nil_expr(rhs))
  {
    assign_top(lhs);
    return;
  }

  const expr2tc r = strip_casts(rhs);

  // A pointer cast into an integer object (`x = (long)p`) leaves this model:
  // the integer arithmetic that may follow is not tracked, so whatever is cast
  // back out of x later may name any object.  Dropping the assignment would
  // leave x empty, and `q = (void *)x` would then copy that empty set.
  if (!may_carry_pointer(lhs->type))
  {
    if (may_carry_pointer(r->type))
      assign_top(lhs);
    return;
  }

  // Field- and index-insensitivity means the destination of `s.f = q` and
  // `p->f = q` is decided entirely by the base object.
  const expr2tc target = base_object(lhs);

  if (is_dereference2t(target))
  {
    add_constraint(
      constraint_kindt::STORE,
      get_node(base_object(to_dereference2t(target).value)),
      eval_rhs(rhs, loc));
    return;
  }

  const node_id l = get_node(target);

  // The two dominant shapes get a constraint directly rather than through a
  // temporary, which keeps the node count close to the variable count.
  if (is_address_of2t(r) && !is_dereference2t(to_address_of2t(r).ptr_obj))
  {
    add_constraint(
      constraint_kindt::ADDRESS_OF,
      l,
      get_node(base_object(to_address_of2t(r).ptr_obj)));
    return;
  }

  if (is_dereference2t(r))
  {
    add_constraint(
      constraint_kindt::LOAD,
      l,
      get_node(base_object(to_dereference2t(r).value)));
    return;
  }

  add_constraint(constraint_kindt::COPY, l, eval_rhs(rhs, loc));
}

void andersent::widen_call(
  const expr2tc &ret,
  const std::vector<expr2tc> &arguments,
  unsigned loc)
{
  if (!is_nil_expr(ret) && may_carry_pointer(ret->type))
    assign_top(ret);

  for (const expr2tc &arg : arguments)
    if (!is_nil_expr(arg) && may_carry_pointer(arg->type))
      add_constraint(constraint_kindt::STORE, eval_rhs(arg, loc), top_source());
}

void andersent::bind_call(
  const irep_idt &callee_name,
  const goto_functiont &callee,
  const expr2tc &ret,
  const std::vector<expr2tc> &arguments,
  unsigned loc)
{
  // One node per formal, shared by every call site (context insensitivity).
  // get_node keys symbols by identifier, so the node built here from the
  // signature is the very one the callee's body uses even when the two carry
  // different spellings of the parameter's type.
  const code_type2t &ftype = to_code_type(callee.type);
  const std::size_t argc = std::min(
    {ftype.arguments.size(), ftype.argument_names.size(), arguments.size()});

  // Arguments no formal can be bound to: a nameless parameter, or one passed
  // beyond the signature (varargs, mismatched prototype) and read back out
  // with va_arg.  They reach the callee by a route this frontend cannot see.
  std::vector<expr2tc> unbound(arguments.begin() + argc, arguments.end());

  for (std::size_t i = 0; i < argc; ++i)
  {
    if (!may_carry_pointer(ftype.arguments[i]))
      continue;

    if (ftype.argument_names[i].empty())
    {
      unbound.push_back(arguments[i]);
      continue;
    }

    const node_id formal =
      get_node(symbol2tc(ftype.arguments[i], ftype.argument_names[i]));

    // An entry-point call synthesised by --func passes nil for every argument,
    // leaving the formal unconstrained: it may hold any address.
    if (is_nil_expr(arguments[i]))
      points_to_top(formal);
    else
      add_constraint(
        constraint_kindt::COPY, formal, eval_rhs(arguments[i], loc));
  }

  if (!unbound.empty())
    widen_call(expr2tc(), unbound, loc);

  if (!is_nil_expr(ret) && may_carry_pointer(ret->type))
    handle_assign(ret, return_symbol(callee_name), loc);
}

void andersent::handle_function_call(
  const code_function_call2t &call,
  const goto_functionst &goto_functions,
  unsigned loc)
{
  if (is_symbol2t(call.function))
  {
    const irep_idt callee_name = to_symbol2t(call.function).thename;
    auto it = goto_functions.function_map.find(callee_name);
    if (it != goto_functions.function_map.end() && it->second.body_available)
      bind_call(callee_name, it->second, call.ret, call.operands, loc);
    else
      widen_call(call.ret, call.operands, loc); // bodyless / intrinsic
    return;
  }

  // A call through a function pointer.  Which functions it reaches depends on
  // the very sets being computed, so defer it to resolve_indirect_calls rather
  // than widening to TOP here and throwing that precision away.
  if (is_dereference2t(call.function))
  {
    indirect_callt deferred;
    deferred.function =
      get_node(base_object(to_dereference2t(call.function).value));
    deferred.ret = call.ret;
    deferred.arguments = call.operands;
    deferred.location_number = loc;
    indirect_calls.push_back(std::move(deferred));
    return;
  }

  widen_call(call.ret, call.operands, loc);
}

bool andersent::resolve_indirect_calls(const goto_functionst &goto_functions)
{
  bool changed = false;

  for (indirect_callt &call : indirect_calls)
  {
    // By value: binding a call interns nodes, which can reallocate pts.
    const std::unordered_set<node_id> targets = points_to(call.function);

    const auto widen_once = [&]() {
      if (call.widened)
        return;
      widen_call(call.ret, call.arguments, call.location_number);
      call.widened = true;
      changed = true;
    };

    // No target known, or the pointer may point anywhere: there is no set of
    // callees to bind, so fall back to the conservative treatment.
    if (targets.empty() || targets.count(TOP))
    {
      widen_once();
      continue;
    }

    for (node_id t : targets)
    {
      const expr2tc &target = node_to_expr[t];
      auto it = goto_functions.function_map.end();
      if (is_symbol2t(target))
        it = goto_functions.function_map.find(to_symbol2t(target).thename);

      // A target that is not a function we can see through is unmodelled.
      if (it == goto_functions.function_map.end() || !it->second.body_available)
      {
        widen_once();
        continue;
      }

      if (!call.bound.insert(it->first).second)
        continue;

      bind_call(
        it->first, it->second, call.ret, call.arguments, call.location_number);
      changed = true;
    }
  }

  return changed;
}

void andersent::handle_other(const expr2tc &code, unsigned loc)
{
  // These only read through their operands (compare goto_symext::symex_other),
  // so they create no points-to facts.
  if (is_code_expression2t(code) || is_code_free2t(code))
    return;

  // Inline asm keeps only its source string in IREP2 -- the operands it may
  // write through are not in the IR at all, so there is nothing to widen per
  // object and the whole result has to degrade to TOP.  ESBMC's C frontend
  // lowers asm to a SKIP, so this is reachable only from a goto binary.
  if (is_code_asm2t(code))
  {
    everything_escapes = true;
    return;
  }

  // Anything else may write through the pointers it is handed: a delete
  // running a destructor this walk never sees, or an asprintf() storing a
  // freshly allocated buffer into its argument.  Same treatment as a call to
  // a function without a body.
  std::vector<expr2tc> args;
  code->foreach_operand([&args](const expr2tc &op) { args.push_back(op); });
  widen_call(expr2tc(), args, loc);
}

void andersent::collect_constraints(const goto_functionst &goto_functions)
{
  forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available)
      continue;

    forall_goto_program_instructions (i_it, f_it->second.body)
    {
      if (is_nil_expr(i_it->code))
        continue;

      const unsigned loc = i_it->location_number;

      if (i_it->is_assign())
      {
        const code_assign2t &assign = to_code_assign2t(i_it->code);
        handle_assign(assign.target, assign.source, loc);
      }
      else if (i_it->is_return())
      {
        const expr2tc &value = to_code_return2t(i_it->code).operand;
        if (!is_nil_expr(value) && may_carry_pointer(value->type))
          add_constraint(
            constraint_kindt::COPY,
            return_node(f_it->first),
            eval_rhs(value, loc));
      }
      else if (i_it->is_function_call())
      {
        handle_function_call(
          to_code_function_call2t(i_it->code), goto_functions, loc);
      }
      else if (i_it->is_other())
      {
        handle_other(i_it->code, loc);
      }
    }
  }
}

void andersent::operator()(const goto_functionst &goto_functions)
{
  collect_constraints(goto_functions);
  solve();

  // A call through a function pointer can only be lowered once that pointer's
  // set is known, and binding its arguments can in turn grow other sets (and
  // so reach further call targets).  Alternate until a round adds nothing;
  // this terminates because each site binds a given callee at most once.
  while (resolve_indirect_calls(goto_functions))
    solve();
}

void andersent::to_object_descriptors(node_id n, valuest &dest) const
{
  const auto &set = points_to(n);

  // TOP subsumes everything: report a single unnameable target so consumers
  // (is_object_descriptor2t checks) abstain / havoc conservatively.
  if (everything_escapes || set.count(TOP))
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

void andersent::get_values(locationt, const expr2tc &expr, valuest &dest)
{
  const node_id *n = find_node(base_object(expr));
  if (n == nullptr)
  {
    dest.push_back(unknown2tc(pointer_type2()));
    return;
  }

  to_object_descriptors(*n, dest);
}

void andersent::get_reference_set(locationt, const expr2tc &expr, valuest &dest)
{
  // The objects `*p`, `p->f` and `p[i]` refer to are exactly pts[p], so reduce
  // to the base access first and then step through the dereference.
  expr2tc pointer = base_object(expr);
  if (is_dereference2t(pointer))
    pointer = base_object(to_dereference2t(pointer).value);

  const node_id *n = find_node(pointer);
  if (n == nullptr)
  {
    // An expression with no node -- a nested dereference such as `(*q)->f`,
    // say -- was never constrained, so nothing is known about it. Returning
    // an empty set here would read as "refers to no object at all" and let
    // consumers keep facts this access invalidates.
    dest.push_back(unknown2tc(pointer_type2()));
    return;
  }

  to_object_descriptors(*n, dest);
}
