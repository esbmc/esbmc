#include <goto-programs/goto_k_induction_legacy.h>
#include <goto-programs/goto_k_induction.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/loopst.h>
#include <irep2/irep2_expr.h>
#include <pointer-analysis/andersen.h>
#include <util/message/message.h>
#include <unordered_map>

namespace
{
bool indexes_through_pointer(const expr2tc &expr)
{
  if (is_dereference2t(expr))
    return true;
  if (is_index2t(expr))
    return indexes_through_pointer(to_index2t(expr).source_value);
  if (is_member2t(expr))
    return indexes_through_pointer(to_member2t(expr).source_value);
  if (is_typecast2t(expr))
    return indexes_through_pointer(to_typecast2t(expr).from);
  if (is_symbol2t(expr))
    return is_pointer_type(expr->type);
  return false;
}

expr2tc extract_queried_pointer(const expr2tc &expr)
{
  if (is_dereference2t(expr))
    return to_dereference2t(expr).value;
  if (is_index2t(expr))
    return extract_queried_pointer(to_index2t(expr).source_value);
  if (is_member2t(expr))
    return extract_queried_pointer(to_member2t(expr).source_value);
  if (is_typecast2t(expr))
    return extract_queried_pointer(to_typecast2t(expr).from);
  if (is_symbol2t(expr) && is_pointer_type(expr->type))
    return expr;
  return expr2tc();
}

struct loop_writest
{
  bool modifies_pointer_array = false;
  bool unresolvable = false;
  loopst::loop_varst ptrs;
};

struct callee_writest
{
  bool writes_through_pointer = false;
  bool modifies_pointer_array = false;
};

class legacy_scant
{
public:
  explicit legacy_scant(const goto_functionst &goto_functions)
    : goto_functions(goto_functions)
  {
  }

  loop_writest scan(const loopst &loop)
  {
    loop_writest w;
    auto it = loop.get_original_loop_head();
    const auto end = loop.get_original_loop_exit();
    for (;; ++it)
    {
      if (it->is_assign())
        lhs(to_code_assign2t(it->code).target, w);
      else if (it->is_function_call())
        call(to_code_function_call2t(it->code), w);
      if (it == end)
        break;
    }
    return w;
  }

private:
  const goto_functionst &goto_functions;
  std::unordered_map<irep_idt, callee_writest, irep_id_hash> summaries;

  void lhs(const expr2tc &expr, loop_writest &w)
  {
    if (is_nil_expr(expr))
      return;
    if (is_dereference2t(expr))
    {
      const expr2tc ptr = extract_queried_pointer(expr);
      if (is_nil_expr(ptr))
        w.unresolvable = true;
      else
        w.ptrs.insert(ptr);
      return;
    }
    if (is_index2t(expr))
    {
      const index2t &idx = to_index2t(expr);
      if (indexes_through_pointer(idx.source_value))
      {
        w.modifies_pointer_array = true;
        const expr2tc ptr = extract_queried_pointer(idx.source_value);
        if (is_nil_expr(ptr))
          w.unresolvable = true;
        else
          w.ptrs.insert(ptr);
      }
      lhs(idx.source_value, w);
      return;
    }
    expr->foreach_operand([this, &w](const expr2tc &e) { lhs(e, w); });
  }

  void call(const code_function_call2t &c, loop_writest &w)
  {
    if (is_dereference2t(c.function))
    {
      w.unresolvable = true;
      return;
    }
    lhs(c.ret, w);
    if (!is_symbol2t(c.function))
      return;
    const callee_writest s = summary(to_symbol2t(c.function).thename);
    if (s.writes_through_pointer)
      for (const expr2tc &arg : c.operands)
        if (!is_nil_expr(arg) && is_pointer_type(arg->type))
          w.ptrs.insert(arg);
    if (s.modifies_pointer_array)
    {
      w.modifies_pointer_array = true;
      w.unresolvable = true;
    }
  }

  static bool modifies_pointer_array(const expr2tc &expr)
  {
    if (is_nil_expr(expr) || is_dereference2t(expr))
      return false;
    if (is_index2t(expr))
      return indexes_through_pointer(to_index2t(expr).source_value) ||
             modifies_pointer_array(to_index2t(expr).source_value);
    bool found = false;
    expr->foreach_operand(
      [&found](const expr2tc &e) { found |= modifies_pointer_array(e); });
    return found;
  }

  callee_writest summary(const irep_idt &name)
  {
    auto cached = summaries.find(name);
    if (cached != summaries.end())
      return cached->second;
    // Placeholder breaks recursion.
    summaries[name] = callee_writest();

    callee_writest s;
    auto f = goto_functions.function_map.find(name);
    if (f != goto_functions.function_map.end() && f->second.body_available)
      for (const auto &instr : f->second.body.instructions)
      {
        expr2tc target;
        if (instr.is_assign())
          target = to_code_assign2t(instr.code).target;
        else if (instr.is_function_call())
        {
          const code_function_call2t &c = to_code_function_call2t(instr.code);
          target = c.ret;
          if (is_symbol2t(c.function))
          {
            const callee_writest inner =
              summary(to_symbol2t(c.function).thename);
            s.writes_through_pointer |= inner.writes_through_pointer;
            s.modifies_pointer_array |= inner.modifies_pointer_array;
          }
        }
        if (is_nil_expr(target))
          continue;
        if (!is_symbol2t(target) && indexes_through_pointer(target))
          s.writes_through_pointer = true;
        s.modifies_pointer_array |= modifies_pointer_array(target);
      }
    summaries[name] = s;
    return s;
  }
};

bool resolve(loopst &loop, const loop_writest &w, value_setst &points_to)
{
  if (w.unresolvable || w.ptrs.empty())
    return false;

  goto_programt::const_targett loc = loop.get_original_loop_head();
  if (!points_to.has_location(loc))
    return false;

  loopst::loop_varst objects;
  for (const expr2tc &ptr : w.ptrs)
  {
    value_setst::valuest values;
    points_to.get_values(loc, ptr, values);
    if (values.empty())
      return false;
    for (const expr2tc &v : values)
    {
      if (!is_object_descriptor2t(v))
        return false;
      const expr2tc &object = to_object_descriptor2t(v).object;
      if (!is_symbol2t(object) || !check_var_name(object))
        return false;
      objects.insert(object);
    }
  }

  for (const expr2tc &obj : objects)
    loop.add_modified_var_to_loop(obj);
  return true;
}

bool calls_nondet_memory(const goto_functionst &goto_functions)
{
  forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available || it->second.body.hide)
      continue;
    for (const auto &instr : it->second.body.instructions)
      if (
        instr.is_function_call() &&
        is_symbol2t(to_code_function_call2t(instr.code).function) &&
        to_symbol2t(to_code_function_call2t(instr.code).function).thename ==
          "c:@F@__VERIFIER_nondet_memory")
        return true;
  }
  return false;
}
} // namespace

bool goto_k_induction_legacy(
  goto_functionst &goto_functions,
  bool continue_past_failed_assertions)
{
  goto_functions.update();
  andersent points_to;
  points_to(goto_functions);
  legacy_scant scan(goto_functions);

  const bool nondet_memory = calls_nondet_memory(goto_functions);
  if (nondet_memory)
    log_warning(
      "k-induction cannot generalise the memory __VERIFIER_nondet_memory "
      "havocs. Disabling inductive step");

  bool unresolved_write = false;
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;
    const bool user_function = !it->second.body.hide;
    goto_loopst loops(it->first, goto_functions, it->second);
    for (auto &loop : loops.get_loops())
    {
      if (user_function)
      {
        const loop_writest w = scan.scan(loop);
        if (w.modifies_pointer_array && !resolve(loop, w, points_to))
          unresolved_write = true;
      }

      if (loop.get_modified_loop_vars().empty())
        continue;
      k_induction_transform_loop(
        it->second, loop, continue_past_failed_assertions);
    }
  }
  goto_functions.update();

  if (unresolved_write)
    log_warning(
      "k-induction cannot havoc an array element a loop writes through a "
      "pointer. Disabling inductive step");
  return nondet_memory || unresolved_write;
}
