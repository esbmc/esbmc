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
      // A nondet value is not an address of anything we can name; it also
      // cannot alias an existing object, so an empty set stays sound.
      return fresh_node();

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

void andersent::handle_assign(
  const expr2tc &lhs,
  const expr2tc &rhs,
  unsigned loc)
{
  if (is_nil_expr(lhs))
    return;

  // A nil source is an unconstrained value, so route it through the node that
  // already points anywhere rather than dropping the assignment.
  if (is_nil_expr(rhs))
  {
    top_source();
    handle_assign(lhs, top_source_expr(), loc);
    return;
  }

  if (!may_carry_pointer(lhs->type) && !may_carry_pointer(rhs->type))
    return;

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
  const expr2tc r = strip_casts(rhs);

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
    handle_assign(ret, top_source_expr(), loc);

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
  const code_type2t &ftype = to_code_type(callee.type);
  const std::size_t argc = std::min(
    {ftype.arguments.size(), ftype.argument_names.size(), arguments.size()});
  for (std::size_t i = 0; i < argc; ++i)
  {
    if (!may_carry_pointer(ftype.arguments[i]))
      continue;

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

void andersent::collect_constraints(const goto_functionst &goto_functions)
{
  // Known gap: inline asm can write memory through operands this frontend
  // never sees, so its effects are not modelled at all.
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

void andersent::get_values(locationt, const expr2tc &expr, valuest &dest)
{
  auto it = expr_to_node.find(base_object(expr));
  if (it != expr_to_node.end())
    to_object_descriptors(it->second, dest);
}

void andersent::get_reference_set(locationt, const expr2tc &expr, valuest &dest)
{
  // The objects `*p`, `p->f` and `p[i]` refer to are exactly pts[p], so reduce
  // to the base access first and then step through the dereference.
  expr2tc pointer = base_object(expr);
  if (is_dereference2t(pointer))
    pointer = base_object(to_dereference2t(pointer).value);

  auto it = expr_to_node.find(pointer);
  if (it != expr_to_node.end())
    to_object_descriptors(it->second, dest);
}
