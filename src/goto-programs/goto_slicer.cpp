#include <goto-programs/goto_slicer.h>

#include <irep2/irep2_utils.h>
#include <pointer-analysis/andersen.h>
#include <util/base/prefix.h>
#include <util/config/options.h>
#include <util/message/message.h>

#include <cassert>

#include <algorithm>
#include <functional>
#include <unordered_map>

namespace
{
using slice_loct = unsigned;

struct accesst
{
  std::vector<slice_loct> locations;
  /// Reaches any address-taken object or allocation site.
  bool top = false;

  void add(slice_loct l)
  {
    locations.push_back(l);
  }

  void compact()
  {
    std::sort(locations.begin(), locations.end());
    locations.erase(
      std::unique(locations.begin(), locations.end()), locations.end());
    locations.shrink_to_fit();
  }
};

/// The precision seam: the abstract locations an expression reads or writes.
/// Whole-object: a member, element or byte of an object is the object, and a
/// dereference reaches every object Andersen says the pointer may point to.
class locationst
{
public:
  locationst(andersent &points_to, const goto_functionst &goto_functions)
    : points_to(points_to)
  {
    std::vector<expr2tc> objects = points_to.addressed_objects();
    // Andersen records an address only where it can follow it; one that
    // escapes into a value it gives up on (a double, say) is still taken.
    std::function<void(const expr2tc &)> taken = [&](const expr2tc &e) {
      if (is_nil_expr(e))
        return;
      if (is_address_of2t(e))
      {
        expr2tc obj = to_address_of2t(e).ptr_obj;
        while (is_member2t(obj) ||
               (is_index2t(obj) && !is_pointer_index(obj)) ||
               is_typecast2t(obj))
          obj = is_member2t(obj)  ? to_member2t(obj).source_value
                : is_index2t(obj) ? to_index2t(obj).source_value
                                  : to_typecast2t(obj).from;
        if (is_symbol2t(obj))
          objects.push_back(obj);
      }
      e->foreach_operand(taken);
    };
    forall_goto_functions (f, goto_functions)
      forall_goto_program_instructions (i, f->second.body)
      {
        taken(i->code);
        taken(i->guard);
      }

    for (const expr2tc &obj : objects)
    {
      top.push_back(object(obj));
      if (is_symbol2t(obj))
        top_names.push_back(to_symbol2t(obj).thename);
    }
    std::sort(top_names.begin(), top_names.end());
    top_names.erase(
      std::unique(top_names.begin(), top_names.end()), top_names.end());
    std::sort(top.begin(), top.end());
    top.erase(std::unique(top.begin(), top.end()), top.end());
    in_top.resize(count);
    for (slice_loct l : top)
      in_top[l] = true;
  }

  /// Members of TOP.
  std::vector<slice_loct> top;
  /// The named ones, functions included.
  std::vector<irep_idt> top_names;
  std::vector<bool> in_top;

  slice_loct size() const
  {
    return count;
  }

  slice_loct named(const irep_idt &id)
  {
    const auto it = by_name.try_emplace(id, count);
    if (it.second)
    {
      // ESBMC's model state (__ESBMC_alloc, ...), which symex reads in
      // checks no instruction spells out.
      if (has_prefix(id.as_string(), "c:@__ESBMC_"))
        implicit.push_back(count);
      ++count;
    }
    return it.first->second;
  }

  /// Read by symex itself, so relevant under every criterion.
  std::vector<slice_loct> implicit;

  void reads(const expr2tc &e, accesst &out)
  {
    if (is_nil_expr(e))
      return;
    if (is_symbol2t(e))
    {
      if (!is_code_type(e->type))
        out.add(named(to_symbol2t(e).thename));
      return;
    }
    if (is_address_of2t(e))
    {
      address_reads(to_address_of2t(e).ptr_obj, out);
      return;
    }
    if (is_dereference2t(e))
    {
      reads(to_dereference2t(e).value, out);
      pointees(to_dereference2t(e).value, out);
      return;
    }
    if (is_pointer_index(e))
    {
      const index2t &index = to_index2t(e);
      reads(index.source_value, out);
      reads(index.index, out);
      pointees(index.source_value, out);
      return;
    }
    if (is_code_printf2t(e))
    {
      // %s and friends read what their pointer operands point to.
      for (const expr2tc &op : to_code_printf2t(e).operands)
      {
        reads(op, out);
        if (is_pointer_type(op->type))
          pointees(op, out);
      }
      return;
    }
    e->foreach_operand([&](const expr2tc &op) { reads(op, out); });
  }

  /// What evaluating the address of the l-value \p e reads.
  void address_reads(const expr2tc &e, accesst &out)
  {
    if (is_symbol2t(e) || is_constant_string2t(e))
      return;
    if (is_dereference2t(e))
      reads(to_dereference2t(e).value, out);
    else if (is_pointer_index(e))
    {
      reads(to_index2t(e).source_value, out);
      reads(to_index2t(e).index, out);
    }
    else if (is_index2t(e))
    {
      address_reads(to_index2t(e).source_value, out);
      reads(to_index2t(e).index, out);
    }
    else if (is_member2t(e))
      address_reads(to_member2t(e).source_value, out);
    else if (is_typecast2t(e))
      address_reads(to_typecast2t(e).from, out);
    else
      reads(e, out);
  }

  void writes(const expr2tc &lhs, accesst &defs, accesst &uses)
  {
    if (is_symbol2t(lhs))
      defs.add(named(to_symbol2t(lhs).thename));
    else if (is_member2t(lhs))
      writes(to_member2t(lhs).source_value, defs, uses);
    else if (is_pointer_index(lhs))
    {
      pointees(to_index2t(lhs).source_value, defs);
      reads(to_index2t(lhs).source_value, uses);
      reads(to_index2t(lhs).index, uses);
    }
    else if (is_index2t(lhs))
    {
      writes(to_index2t(lhs).source_value, defs, uses);
      reads(to_index2t(lhs).index, uses);
    }
    else if (is_dereference2t(lhs))
    {
      pointees(to_dereference2t(lhs).value, defs);
      reads(to_dereference2t(lhs).value, uses);
    }
    else if (is_typecast2t(lhs))
      writes(to_typecast2t(lhs).from, defs, uses);
    else if (is_bitcast2t(lhs))
      writes(to_bitcast2t(lhs).from, defs, uses);
    else if (is_byte_extract2t(lhs))
    {
      writes(to_byte_extract2t(lhs).source_value, defs, uses);
      reads(to_byte_extract2t(lhs).source_offset, uses);
    }
    else if (is_if2t(lhs))
    {
      reads(to_if2t(lhs).cond, uses);
      writes(to_if2t(lhs).true_value, defs, uses);
      writes(to_if2t(lhs).false_value, defs, uses);
    }
    else
    {
      defs.top = true;
      reads(lhs, uses);
    }
  }

  /// What symex's dereference checks in \p e read: the pointer and index
  /// operands of each dereference, not the values loaded.
  void check_reads(const expr2tc &e, accesst &out)
  {
    if (is_nil_expr(e))
      return;
    if (is_dereference2t(e))
      reads(to_dereference2t(e).value, out);
    else if (is_pointer_index(e))
    {
      reads(to_index2t(e).source_value, out);
      reads(to_index2t(e).index, out);
    }
    else if (is_index2t(e))
    {
      // Inside a dereferenced object the index is part of the offset checked.
      reads(to_index2t(e).index, out);
      check_reads(to_index2t(e).source_value, out);
    }
    else if (is_byte_extract2t(e))
    {
      reads(to_byte_extract2t(e).source_offset, out);
      check_reads(to_byte_extract2t(e).source_value, out);
    }
    else
      e->foreach_operand([&](const expr2tc &op) { check_reads(op, out); });
  }

  /// The functions a call through \p pointer may reach; nullopt if unknown.
  std::optional<std::vector<irep_idt>> targets(const expr2tc &pointer)
  {
    value_setst::valuest values;
    points_to.get_values(goto_programt::const_targett(), pointer, values);
    std::vector<irep_idt> result;
    for (const expr2tc &v : values)
    {
      if (!is_object_descriptor2t(v))
        return std::nullopt;
      const expr2tc &obj = to_object_descriptor2t(v).object;
      if (is_symbol2t(obj))
        result.push_back(to_symbol2t(obj).thename);
      else if (!andersent::is_nondet_object(obj))
        return std::nullopt;
    }
    return result;
  }

  /// The objects \p pointer may point to.
  void pointees(const expr2tc &pointer, accesst &out)
  {
    value_setst::valuest values;
    points_to.get_values(goto_programt::const_targett(), pointer, values);
    for (const expr2tc &v : values)
    {
      if (!is_object_descriptor2t(v))
        out.top = true;
      // Symex resolves a dereference of an unconstrained pointer to an
      // invalid object, which no other access reaches.
      else if (!andersent::is_nondet_object(to_object_descriptor2t(v).object))
        out.add(object(to_object_descriptor2t(v).object));
    }
  }
  static bool is_pointer_index(const expr2tc &e)
  {
    return is_index2t(e) && is_pointer_type(to_index2t(e).source_value->type);
  }

  static bool dereferences(const expr2tc &e)
  {
    if (is_nil_expr(e))
      return false;
    if (is_dereference2t(e) || is_pointer_index(e))
      return true;
    bool found = false;
    e->foreach_operand(
      [&](const expr2tc &op) { found = found || dereferences(op); });
    return found;
  }

private:
  andersent &points_to;
  std::unordered_map<irep_idt, slice_loct, irep_id_hash> by_name;
  std::unordered_map<expr2tc, slice_loct, irep2_hash> by_expr;
  slice_loct count = 0;

  slice_loct object(const expr2tc &obj)
  {
    if (is_symbol2t(obj))
      return named(to_symbol2t(obj).thename);
    const auto it = by_expr.try_emplace(obj, count);
    if (it.second)
      ++count;
    return it.first->second;
  }
};

bool contains_nondet(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (
    is_sideeffect2t(e) &&
    to_sideeffect2t(e).kind == sideeffect2t::allockind::nondet)
    return true;
  bool found = false;
  e->foreach_operand(
    [&](const expr2tc &op) { found = found || contains_nondet(op); });
  return found;
}

bool is_allocation(const expr2tc &e)
{
  expr2tc r = e;
  while (is_typecast2t(r))
    r = to_typecast2t(r).from;
  if (!is_sideeffect2t(r))
    return false;
  switch (to_sideeffect2t(r).kind)
  {
  case sideeffect2t::allockind::malloc:
  case sideeffect2t::allockind::realloc:
  case sideeffect2t::allockind::alloca:
  case sideeffect2t::allockind::cpp_new:
  case sideeffect2t::allockind::cpp_new_arr:
    return true;
  default:
    return false;
  }
}

enum : uint8_t
{
  NOT_KEPT = 0,
  CHECK_ONLY = 1,
  FULL = 2
};
} // namespace

struct goto_slicert::programt
{
  struct nodet
  {
    const goto_programt::instructiont *instruction;
    unsigned function;
    /// Where a GOTO jumps; a RETURN jumps to its END_FUNCTION.
    std::optional<unsigned> target;
    accesst defs, uses, check_uses;
    /// Not a SKIP, LOCATION or other no-op, so it counts in the report.
    bool counted = false;
    /// Kept under every criterion.
    bool always = false;
    bool dereferences = false;
    bool allocates = false;
    bool frees = false;
    bool nondet = false;
    /// DECL or DEAD of an address-taken variable.
    bool lifetime = false;
    /// An assignment whose value may carry an address.
    bool stores_address = false;
    /// Writes or ends a variable that may hold an address: overwriting the
    /// last reference to an object leaks it.
    bool holds_address = false;
    /// A backward GOTO, or a jump out of a loop: what decides when it exits.
    bool loop_control = false;
    /// An exit test at the head of a collapsible loop.
    bool head = false;
  };

  struct functiont
  {
    unsigned begin, end;
    std::vector<unsigned> callers;
  };

  struct loopt
  {
    unsigned head, body, backedge;
    /// Only one backward GOTO reaches the head.
    bool collapsible;
  };

  std::vector<nodet> nodes;
  std::vector<functiont> functions;
  std::vector<loopt> loops;
  std::vector<std::vector<unsigned>> definers;
  /// Instructions that may write any address-taken object.
  std::vector<unsigned> top_definers;
  std::vector<slice_loct> top;
  std::vector<bool> in_top;
  std::vector<slice_loct> implicit;
};

namespace
{
using programt = goto_slicert::programt;
using nodet = programt::nodet;

struct slice_statet
{
  const programt &p;
  const bool drop_inductive_step;
  const bool collapse_loops;
  std::vector<uint8_t> kept;
  std::vector<bool> relevant;
  std::vector<bool> live;
  bool top_relevant = false;
  bool top_read = false;
  std::vector<unsigned> work;

  slice_statet(const programt &p, bool drop_inductive_step, bool collapse_loops)
    : p(p),
      drop_inductive_step(drop_inductive_step),
      collapse_loops(collapse_loops),
      kept(p.nodes.size(), NOT_KEPT),
      relevant(p.definers.size(), false),
      live(p.functions.size(), false)
  {
  }

  bool excluded(unsigned i) const
  {
    return drop_inductive_step &&
           p.nodes[i].instruction->inductive_step_instruction;
  }

  void keep(unsigned i, uint8_t level)
  {
    if (excluded(i) || kept[i] >= level)
      return;
    kept[i] = level;
    work.push_back(i);
  }

  void mark(slice_loct l)
  {
    if (relevant[l])
      return;
    relevant[l] = true;
    for (unsigned i : p.definers[l])
      keep(i, FULL);
    if (p.in_top[l] && !top_relevant)
    {
      top_relevant = true;
      for (unsigned i : p.top_definers)
        keep(i, FULL);
    }
  }

  void mark(const accesst &a)
  {
    for (slice_loct l : a.locations)
      mark(l);
    if (a.top && !top_read)
    {
      top_read = true;
      for (slice_loct l : p.top)
        mark(l);
    }
  }

  void drain()
  {
    while (!work.empty())
    {
      const unsigned i = work.back();
      work.pop_back();
      const nodet &n = p.nodes[i];
      mark(n.check_uses);
      if (kept[i] == FULL)
        mark(n.uses);
      if (!live[n.function])
      {
        live[n.function] = true;
        for (unsigned call : p.functions[n.function].callers)
          keep(call, FULL);
      }
    }
  }

  /// Kept instructions in [from, to) of one function, from a prefix count.
  static bool any(
    const std::vector<unsigned> &prefix,
    unsigned base,
    unsigned from,
    unsigned to)
  {
    return from < to && prefix[to - base] != prefix[from - base];
  }

  /// One round of the control-flow rules; true iff it kept anything.
  bool control_flow()
  {
    bool changed = false;
    std::vector<unsigned> prefix;
    std::vector<std::vector<unsigned>> prefixes(p.functions.size());
    for (unsigned f = 0; f < p.functions.size(); ++f)
    {
      const programt::functiont &fn = p.functions[f];
      prefix.assign(fn.end - fn.begin + 1, 0);
      for (unsigned i = fn.begin; i < fn.end; ++i)
        prefix[i - fn.begin + 1] = prefix[i - fn.begin] + (kept[i] != NOT_KEPT);

      // A forward jump matters iff something it may skip is kept. Backward
      // ones are loop control, kept from the start.
      for (unsigned i = fn.begin; i < fn.end; ++i)
      {
        const nodet &n = p.nodes[i];
        if (
          n.target && *n.target > i && !n.head && kept[i] == NOT_KEPT &&
          any(prefix, fn.begin, i + 1, *n.target))
        {
          keep(i, FULL);
          changed = true;
        }
      }
      prefixes[f].swap(prefix);
    }

    for (const programt::loopt &loop : p.loops)
    {
      const programt::functiont &fn = p.functions[p.nodes[loop.head].function];
      const auto &pre = prefixes[p.nodes[loop.head].function];
      if (collapse_loops && !any(pre, fn.begin, loop.body, loop.backedge))
        continue;
      for (unsigned h = loop.head; h < loop.body; ++h)
        if (kept[h] != FULL && !excluded(h))
        {
          keep(h, FULL);
          changed = true;
        }
    }
    return changed;
  }

  bool collapsed(const programt::loopt &loop) const
  {
    if (!collapse_loops || !loop.collapsible)
      return false;
    for (unsigned i = loop.body; i < loop.backedge; ++i)
      if (kept[i] != NOT_KEPT)
        return false;
    return true;
  }
};

class buildert
{
public:
  buildert(
    programt &p,
    const goto_functionst &goto_functions,
    andersent &points_to)
    : p(p),
      goto_functions(goto_functions),
      points_to(points_to),
      locations(points_to, goto_functions)
  {
  }

  void build()
  {
    discover_functions();
    for (unsigned f = 0; f < names.size(); ++f)
      flatten(f);
    for (unsigned i = 0; i < p.nodes.size(); ++i)
      describe(i);
    find_loops();
    keep_recursive_calls();
    index_definers();
  }

private:
  programt &p;
  const goto_functionst &goto_functions;
  const andersent &points_to;
  locationst locations;

  std::vector<irep_idt> names;
  std::unordered_map<irep_idt, unsigned, irep_id_hash> index;
  std::unordered_map<const goto_programt::instructiont *, unsigned> position;
  /// Call targets of each node, filled for FUNCTION_CALLs.
  std::vector<std::vector<unsigned>> callees;
  /// Every function with a body whose address is taken.
  std::optional<std::vector<irep_idt>> addressed_functions;

  const goto_functiont *body(const irep_idt &name) const
  {
    const auto it = goto_functions.function_map.find(name);
    if (it == goto_functions.function_map.end() || !it->second.body_available)
      return nullptr;
    return &it->second;
  }

  const std::vector<irep_idt> &all_addressed_functions()
  {
    if (!addressed_functions)
    {
      addressed_functions.emplace();
      for (const irep_idt &name : locations.top_names)
        if (body(name))
          addressed_functions->push_back(name);
    }
    return *addressed_functions;
  }

  /// The functions \p call may reach; nullopt if any function may be.
  std::optional<std::vector<irep_idt>>
  call_targets(const code_function_call2t &call)
  {
    if (is_symbol2t(call.function))
      return std::vector<irep_idt>{to_symbol2t(call.function).thename};
    if (is_dereference2t(call.function))
      return locations.targets(to_dereference2t(call.function).value);
    return std::nullopt;
  }

  void discover_functions()
  {
    std::vector<irep_idt> work{goto_functions.main_id()};
    while (!work.empty())
    {
      const irep_idt name = work.back();
      work.pop_back();
      const goto_functiont *function = body(name);
      if (!function || !index.emplace(name, names.size()).second)
        continue;
      names.push_back(name);
      forall_goto_program_instructions (it, function->body)
      {
        if (!it->is_function_call())
          continue;
        const auto targets = call_targets(to_code_function_call2t(it->code));
        const auto &reached = targets ? *targets : all_addressed_functions();
        work.insert(work.end(), reached.begin(), reached.end());
      }
    }
  }

  void flatten(unsigned f)
  {
    const goto_programt &body = goto_functions.function_map.at(names[f]).body;
    programt::functiont fn;
    fn.begin = p.nodes.size();
    forall_goto_program_instructions (it, body)
    {
      position.emplace(&*it, p.nodes.size());
      programt::nodet n;
      n.instruction = &*it;
      n.function = f;
      p.nodes.push_back(std::move(n));
    }
    fn.end = p.nodes.size();
    p.functions.push_back(std::move(fn));
  }

  slice_loct return_location(unsigned f)
  {
    return locations.named("goto_slice::return::" + names[f].as_string());
  }

  void describe(unsigned i)
  {
    nodet &n = p.nodes[i];
    const goto_programt::instructiont &instr = *n.instruction;
    n.counted =
      !(instr.is_skip() || instr.is_location() || instr.is_end_function() ||
        instr.is_atomic_begin() || instr.is_atomic_end());
    n.dereferences = locationst::dereferences(instr.code) ||
                     locationst::dereferences(instr.guard);
    locations.check_reads(instr.code, n.check_uses);
    locations.check_reads(instr.guard, n.check_uses);

    switch (instr.type)
    {
    case GOTO:
      n.target = position.at(&*instr.targets.front());
      [[fallthrough]];
    case ASSUME:
    case ASSERT:
      locations.reads(instr.guard, n.uses);
      break;

    case ASSIGN:
    {
      const code_assign2t &assign = to_code_assign2t(instr.code);
      locations.writes(assign.target, n.defs, n.uses);
      locations.reads(assign.source, n.uses);
      describe_printf_writes(assign.source, n);
      n.allocates = is_allocation(assign.source);
      n.frees = n.allocates; // realloc
      n.nondet = contains_nondet(assign.source);
      // A check-only store keeps its value when it may carry an address.
      n.stores_address = points_to.carries_address(assign.source);
      n.holds_address =
        n.stores_address || points_to.carries_address(assign.target);
      if (n.stores_address)
        locations.reads(assign.source, n.check_uses);
      break;
    }

    case DECL:
    case DEAD:
    {
      const irep_idt &name = instr.is_decl() ? to_code_decl2t(instr.code).value
                                             : to_code_dead2t(instr.code).value;
      const slice_loct l = locations.named(name);
      n.defs.add(l);
      n.lifetime = l < locations.in_top.size() && locations.in_top[l];
      n.holds_address =
        points_to.carries_address(symbol2tc(pointer_type2(), name));
      break;
    }

    case RETURN:
      n.target = p.functions[n.function].end - 1;
      n.defs.add(return_location(n.function));
      locations.reads(to_code_return2t(instr.code).operand, n.uses);
      break;

    case FUNCTION_CALL:
      describe_call(i);
      break;

    case OTHER:
      if (
        is_code_free2t(instr.code) || is_code_cpp_delete2t(instr.code) ||
        is_code_cpp_del_array2t(instr.code))
      {
        n.frees = true;
        locations.reads(instr.code, n.uses);
      }
      else if (is_code_expression2t(instr.code) || is_code_printf2t(instr.code))
      {
        locations.reads(instr.code, n.uses);
        describe_printf_writes(instr.code, n);
      }
      else
        n.always = !is_nil_expr(instr.code);
      n.counted = n.counted && !is_nil_expr(instr.code);
      break;

    case SKIP:
    case LOCATION:
    case END_FUNCTION:
    case ATOMIC_BEGIN:
    case ATOMIC_END:
      break;

    default:
      n.always = true;
    }
  }

  /// sprintf and the rest of the family but printf write through their
  /// pointer operands; symex runs them, so they are always kept.
  void describe_printf_writes(const expr2tc &e, nodet &n)
  {
    if (!is_code_printf2t(e))
      return;
    const code_printf2t &printf = to_code_printf2t(e);
    if (
      printf.kind == printf_kindt::PRINTF ||
      printf.kind == printf_kindt::VPRINTF)
      return;
    n.always = true;
    for (const expr2tc &op : printf.operands)
      if (is_pointer_type(op->type))
        locations.pointees(op, n.defs);
  }

  void describe_call(unsigned i)
  {
    nodet &n = p.nodes[i];
    const code_function_call2t &call =
      to_code_function_call2t(n.instruction->code);
    if (!is_nil_expr(call.ret))
    {
      n.holds_address = points_to.carries_address(call.ret);
      locations.writes(call.ret, n.defs, n.uses);
    }
    for (const expr2tc &arg : call.operands)
      locations.reads(arg, n.uses);
    if (!is_symbol2t(call.function))
      locations.reads(call.function, n.uses);

    const auto targets = call_targets(call);
    if (!targets)
      n.always = true;
    const auto &reached = targets ? *targets : all_addressed_functions();

    if (callees.size() <= i)
      callees.resize(p.nodes.size());
    for (const irep_idt &name : reached)
    {
      const auto it = index.find(name);
      if (it == index.end())
      {
        // A bodyless callee: symex gives it a fresh result and runs only the
        // intrinsics, which may write through their arguments.
        if (andersent::symex_runs(name))
        {
          n.always = true;
          for (const expr2tc &arg : call.operands)
            if (!is_nil_expr(arg) && is_pointer_type(arg->type))
            {
              locations.pointees(arg, n.uses);
              locations.pointees(arg, n.defs);
            }
        }
        else if (!is_nil_expr(call.ret))
          n.nondet = true;
        continue;
      }
      const unsigned f = it->second;
      callees[i].push_back(f);
      p.functions[f].callers.push_back(i);
      n.uses.add(return_location(f));
      const goto_functiont &callee = goto_functions.function_map.at(name);
      for (const irep_idt &formal : to_code_type(callee.type).argument_names)
        if (!formal.empty())
          n.defs.add(locations.named(formal));
    }
  }

  void find_loops()
  {
    std::unordered_map<unsigned, unsigned> backedges;
    for (unsigned i = 0; i < p.nodes.size(); ++i)
      if (p.nodes[i].target && *p.nodes[i].target <= i)
        ++backedges[*p.nodes[i].target];

    for (unsigned b = 0; b < p.nodes.size(); ++b)
    {
      nodet &back = p.nodes[b];
      if (!back.target || *back.target > b || !back.instruction->is_goto())
        continue;
      const unsigned head = *back.target;
      back.loop_control = true;

      programt::loopt loop{head, head, b, backedges[head] == 1};
      // Exit tests at the head that leave to just past the loop fold into
      // the assumption that replaces a collapsed loop.
      if (loop.collapsible)
        while (loop.body < b && p.nodes[loop.body].instruction->is_goto() &&
               p.nodes[loop.body].target == b + 1)
          p.nodes[loop.body++].head = true;

      for (unsigned i = loop.body; i < b; ++i)
      {
        nodet &n = p.nodes[i];
        if (n.target && (*n.target < head || *n.target > b))
          n.loop_control = true;
      }
      p.loops.push_back(loop);
    }
  }

  /// Tarjan's SCCs over the call graph: a call into a cycle carries a
  /// recursion bound, so it is always kept.
  void keep_recursive_calls()
  {
    const unsigned size = p.functions.size();
    std::vector<std::vector<unsigned>> edges(size);
    for (unsigned i = 0; i < callees.size(); ++i)
      for (unsigned f : callees[i])
        edges[p.nodes[i].function].push_back(f);

    std::vector<int> order(size, -1), low(size, 0);
    std::vector<bool> on_stack(size, false), recursive(size, false);
    std::vector<unsigned> stack;
    int counter = 0;
    std::function<void(unsigned)> visit = [&](unsigned v) {
      order[v] = low[v] = counter++;
      stack.push_back(v);
      on_stack[v] = true;
      for (unsigned w : edges[v])
      {
        if (order[w] < 0)
        {
          visit(w);
          low[v] = std::min(low[v], low[w]);
        }
        else if (on_stack[w])
          low[v] = std::min(low[v], order[w]);
      }
      if (low[v] != order[v])
        return;
      std::vector<unsigned> component;
      unsigned w;
      do
      {
        w = stack.back();
        stack.pop_back();
        on_stack[w] = false;
        component.push_back(w);
      } while (w != v);
      const bool self_call =
        std::find(edges[v].begin(), edges[v].end(), v) != edges[v].end();
      if (component.size() > 1 || self_call)
        for (unsigned c : component)
          recursive[c] = true;
    };
    for (unsigned v = 0; v < size; ++v)
      if (order[v] < 0)
        visit(v);

    for (unsigned i = 0; i < callees.size(); ++i)
      for (unsigned f : callees[i])
        if (recursive[f])
          p.nodes[i].always = true;
  }

  void index_definers()
  {
    p.definers.assign(locations.size(), {});
    p.top = locations.top;
    p.implicit = locations.implicit;
    p.in_top = locations.in_top;
    p.in_top.resize(locations.size(), false);
    for (unsigned i = 0; i < p.nodes.size(); ++i)
    {
      nodet &n = p.nodes[i];
      n.defs.compact();
      n.uses.compact();
      n.check_uses.compact();
      for (slice_loct l : n.defs.locations)
        p.definers[l].push_back(i);
      if (n.defs.top)
        p.top_definers.push_back(i);
    }
  }
};
} // namespace

namespace
{
/// A check-only store whose value nothing reads and whose right-hand side
/// neither carries an address nor needs its own dereference check.
bool zeroable(const nodet &n)
{
  if (!n.instruction->is_assign() || n.stores_address)
    return false;
  const expr2tc &source = to_code_assign2t(n.instruction->code).source;
  return !locationst::dereferences(source) &&
         (is_number_type(source->type) || is_bool_type(source->type));
}
} // namespace

slice_criteriont slice_criteriont::from_options(
  const optionst &options,
  slice_stept step,
  bool keep_nondets)
{
  slice_criteriont c;
  c.step = step;
  c.pointer_check = !options.get_bool_option("no-pointer-check");
  c.memory_leak_check = options.get_bool_option("memory-leak-check");
  c.no_reachable_memory_leak =
    options.get_bool_option("no-reachable-memory-leak");
  c.no_assertions = options.get_bool_option("no-assertions");
  c.keep_nondets = keep_nondets && step == slice_stept::base_case;
  // The k-step strategies ask a bounded question only through the forward
  // condition, which a collapsed loop answers soundly.
  const bool k_step = options.get_bool_option("k-induction") ||
                      options.get_bool_option("k-induction-parallel") ||
                      options.get_bool_option("incremental-bmc") ||
                      options.get_bool_option("falsification");
  const bool bounded = atol(options.get_option("unwind").c_str()) > 0 ||
                       !options.get_option("unwindset").empty();
  c.collapse_loops =
    k_step || !bounded || options.get_bool_option("no-unwinding-assertions");
  return c;
}

std::string slice_resultt::summary() const
{
  return std::to_string(removed_instructions) + " of " +
         std::to_string(instructions) + " instructions removed (" +
         std::to_string(check_only_instructions) + " kept only for checks), " +
         std::to_string(collapsed_loops) + " of " + std::to_string(loops) +
         " loops collapsed, " + std::to_string(removed_calls) + " of " +
         std::to_string(calls) + " calls removed";
}

goto_slicert::goto_slicert(
  const goto_functionst &goto_functions,
  andersent &points_to)
  : program(std::make_unique<programt>())
{
  buildert(*program, goto_functions, points_to).build();
}

goto_slicert::~goto_slicert() = default;

slice_resultt goto_slicert::slice(const slice_criteriont &c) const
{
  const programt &p = *program;
  slice_statet s(p, c.step != slice_stept::inductive_step, c.collapse_loops);
  for (slice_loct l : p.implicit)
    s.mark(l);

  for (unsigned i = 0; i < p.nodes.size(); ++i)
  {
    const nodet &n = p.nodes[i];
    const goto_programt::instructiont &instr = *n.instruction;
    const bool full =
      n.always || n.loop_control || instr.is_assume() ||
      (instr.is_assert() && !(c.no_assertions && instr.is_user_assertion())) ||
      ((c.pointer_check || c.memory_leak_check) && (n.allocates || n.frees)) ||
      (c.pointer_check && n.lifetime) ||
      (c.memory_leak_check && c.no_reachable_memory_leak && n.holds_address) ||
      (c.keep_nondets && n.nondet);
    if (full)
      s.keep(i, FULL);
    else if (c.pointer_check && n.dereferences)
      s.keep(i, CHECK_ONLY);
    // A head test may fold into the loop's assumption, but its guard decides
    // when the loop exits either way.
    if (n.head && !s.excluded(i))
      s.mark(n.uses);
  }

  do
    s.drain();
  while (s.control_flow());

  slice_resultt result;
  for (unsigned i = 0; i < p.nodes.size(); ++i)
  {
    const nodet &n = p.nodes[i];
    if (!n.counted || s.excluded(i))
      continue;
    const bool call = n.instruction->is_function_call();
    ++result.instructions;
    result.calls += call;
    if (s.kept[i] == NOT_KEPT)
    {
      ++result.removed_instructions;
      result.removed_calls += call;
      result.removed.insert(n.instruction);
    }
    else if (s.kept[i] == CHECK_ONLY)
    {
      ++result.check_only_instructions;
      result.check_only.insert(n.instruction);
      if (zeroable(n))
        result.zeroed.insert(n.instruction);
    }
  }

  for (const programt::loopt &loop : p.loops)
  {
    ++result.loops;
    if (s.collapsed(loop))
    {
      ++result.collapsed_loops;
      // Nothing in the body changes the exit tests, so the loop either exits
      // on its first test or never.
      const goto_programt::instructiont &back =
        *p.nodes[loop.backedge].instruction;
      expr2tc exits = not2tc(back.guard);
      for (unsigned h = loop.head; h < loop.body; ++h)
        exits = or2tc(p.nodes[h].instruction->guard, exits);
      result.collapsed.emplace(&back, exits);
    }
  }
  return result;
}

void apply_slice(
  const goto_functionst &analysed,
  goto_functionst &target,
  const slice_resultt &result)
{
  for (auto &[name, function] : target.function_map)
  {
    const auto &from = analysed.function_map.at(name).body.instructions;
    assert(from.size() == function.body.instructions.size());
    auto source = from.begin();
    for (auto &instruction : function.body.instructions)
    {
      const goto_programt::instructiont *key = &*source++;
      if (
        const auto it = result.collapsed.find(key);
        it != result.collapsed.end())
        instruction.make_assumption(it->second);
      else if (result.removed.count(key))
        instruction.make_skip();
      else if (result.zeroed.count(key))
      {
        code_assign2t &assign = to_code_assign2t(instruction.code);
        assign.source = gen_zero(assign.source->type);
      }
    }
  }
}

goto_slice_cachet::goto_slice_cachet(
  goto_functionst &program,
  const optionst &options,
  andersent &points_to,
  bool keep_nondets,
  bool in_place)
  : original(program),
    options(options),
    keep_nondets(keep_nondets),
    in_place(in_place),
    slicer(std::make_unique<goto_slicert>(program, points_to))
{
}

goto_slice_cachet::~goto_slice_cachet() = default;

const slice_resultt &goto_slice_cachet::result_for(slice_stept step)
{
  // A slice in place changed the program the analysis describes.
  assert(!sliced_in_place);
  if (const auto it = results.find(step); it != results.end())
    return it->second;
  return results
    .emplace(
      step,
      slicer->slice(
        slice_criteriont::from_options(options, step, keep_nondets)))
    .first->second;
}

void goto_slice_cachet::report(const std::vector<slice_stept> &steps)
{
  static const std::map<slice_stept, const char *> names{
    {slice_stept::base_case, "base case"},
    {slice_stept::inductive_step, "inductive step"}};
  for (slice_stept step : steps)
    log_status(
      "GOTO slice ({}): {}", names.at(step), result_for(step).summary());
}

goto_functionst &goto_slice_cachet::program_for(slice_stept step)
{
  if (sliced_in_place)
  {
    if (*sliced_in_place != step)
      throw "--goto-slice: a parallel step asked for another step's program";
    return original;
  }
  if (const auto it = programs.find(step); it != programs.end())
    return it->second;

  const slice_resultt &result = result_for(step);
  if (in_place)
  {
    apply_slice(original, original, result);
    sliced_in_place = step;
    slicer.reset();
    results.clear();
    return original;
  }

  goto_functionst &copy = programs[step];
  for (const auto &[name, function] : original.function_map)
  {
    goto_functiont &to = copy.function_map[name];
    to.type = function.type;
    to.body_available = function.body_available;
    to.exception_spec = function.exception_spec;
    to.inlined_funcs = function.inlined_funcs;
    // copy_from, not copy construction: that renumbers locations.
    to.body.copy_from(function.body);
  }
  apply_slice(original, copy, result);
  return copy;
}
