#include <goto-programs/goto_check_excessive_alloc.h>

#include <util/c_types.h>
#include <util/namespace.h>
#include <util/type_byte_size.h>

namespace
{
/// Human-readable allocator name for an allocation side-effect kind, used in
/// the property comment. Returns nullptr for kinds this pass does not check
/// (scalar `new`, `alloca`, and the non-allocation side-effect kinds): a
/// single object or a stack allocation is not an excessive-heap-size concern.
const char *alloc_fn_name(sideeffect2t::allockind kind)
{
  switch (kind)
  {
  case sideeffect2t::allockind::malloc:
    return "malloc";
  case sideeffect2t::allockind::realloc:
    return "realloc";
  case sideeffect2t::allockind::cpp_new_arr:
    return "operator new[]";
  default:
    return nullptr;
  }
}

/// Total requested allocation size in bytes for `se`, or a nil expression
/// when it cannot be computed (an element type with no static size).
///
/// The allocation model (`goto_symext::symex_mem`, src/goto-symex/
/// builtin_functions/memory_alloc.cpp) uniformly allocates `se.size` elements
/// of `se.alloctype`, so the requested byte count is `size * sizeof(alloctype)`
/// for every kind we check. This matters for typed requests: the frontend
/// lowers a bare `malloc(sizeof(T))` to `size == 1, alloctype == T` (never a
/// raw byte count), so without the scaling a large `T` would be checked as
/// `1 <= K` and slip through. A byte-count request such as `malloc(n)` or
/// `n * sizeof(T)` keeps `alloctype == char` (sizeof 1), so the scaling is the
/// identity there.
expr2tc alloc_byte_size(const sideeffect2t &se, const namespacet &ns)
{
  if (is_nil_expr(se.size))
    return expr2tc();

  expr2tc size = se.size;
  if (size->type != size_type2())
    size = typecast2tc(size_type2(), size);

  // A char/unknown element type is already a byte count: no scaling needed.
  if (is_nil_type(se.alloctype) || is_empty_type(se.alloctype))
    return size;

  try
  {
    // size * sizeof(element), in the same modular size_type arithmetic symex
    // uses to model the allocation. A count near 2^64 can wrap the product to
    // a small value and slip under K — a false negative, never a false
    // positive; it mirrors the model rather than contradicting it.
    BigInt elem_bytes = type_byte_size(se.alloctype, &ns);
    if (elem_bytes == 1)
      return size; // sizeof(char)-equivalent: avoid a redundant `* 1`
    return mul2tc(
      size_type2(), size, constant_int2tc(size_type2(), elem_bytes));
  }
  catch (const array_type2t::array_size_excp &)
  {
    // Element type has no static byte size (flexible/infinite array member).
    return expr2tc();
  }
}

/// One allocation reached from an assignment's right-hand side.
struct alloc_site_t
{
  expr2tc byte_size; ///< total request in bytes
  expr2tc guard;     ///< path condition it is evaluated under (nil == always)
  const char *fn;    ///< allocator name for the property comment
};

/// Collect the allocation side-effects reachable from `e`, threading the
/// branch condition each one is evaluated under. The Clang frontend lowers
/// `realloc(p, n)` to `p == 0 ? malloc(n) : realloc(p, n)`, so allocations
/// can sit inside an `if` expression rather than being the bare right-hand
/// side; guarding each site keeps the reported allocator accurate and avoids
/// flagging the size of a branch that is not taken.
void collect_allocs(
  const expr2tc &e,
  const expr2tc &guard,
  const namespacet &ns,
  std::vector<alloc_site_t> &out)
{
  if (is_nil_expr(e))
    return;

  if (is_sideeffect2t(e))
  {
    const sideeffect2t &se = to_sideeffect2t(e);
    if (const char *fn = alloc_fn_name(se.kind))
    {
      expr2tc bytes = alloc_byte_size(se, ns);
      if (!is_nil_expr(bytes))
        out.push_back({bytes, guard, fn});
    }
    // The size/arguments of an allocation are already flattened temporaries,
    // never nested allocations, so no further recursion is needed here.
    return;
  }

  if (is_if2t(e))
  {
    const if2t &i = to_if2t(e);
    const expr2tc not_cond = not2tc(i.cond);
    collect_allocs(
      i.true_value,
      is_nil_expr(guard) ? i.cond : and2tc(guard, i.cond),
      ns,
      out);
    collect_allocs(
      i.false_value,
      is_nil_expr(guard) ? not_cond : and2tc(guard, not_cond),
      ns,
      out);
    return;
  }

  e->foreach_operand(
    [&](const expr2tc &op) { collect_allocs(op, guard, ns, out); });
}
} // namespace

// This pass instruments *every* function, user code and linked operational
// models alike. malloc/realloc/operator new[] are frontend intrinsics lowered
// to inline sideeffects at the user call site, so they are checked there.
// Model-based allocators (calloc, strdup, kmalloc, ldv_malloc, ...) are real
// functions whose OM bodies allocate via malloc/realloc; instrumenting those
// bodies is what gives calloc & co. their CWE-789 coverage. The trade-off is
// that such a finding is reported at the model's malloc site (e.g.
// stdlib.c:calloc) rather than the user's call site, and a library routine
// that allocates an input-sized buffer (e.g. strdup of a symbolic-length
// string) can surface a genuine — but library-located — CWE-789. Both are
// documented in website/content/docs/cwe-mapping.md.
bool goto_check_excessive_alloc::runOnFunction(
  std::pair<const irep_idt, goto_functiont> &F)
{
  if (!F.second.body_available)
    return false;

  namespacet ns(context);
  goto_programt &body = F.second.body;

  // Collect the target assignments (with their allocation sites) first:
  // inserting an ASSERT before an allocation shifts that allocation to the
  // next slot, so a single forward scan that inserted in place would
  // re-encounter and re-instrument it. Scanning assignment right-hand sides is
  // sufficient: every allocation that survives to the GOTO program lands on an
  // assignment (an allocation whose result is discarded, `malloc(n);`, is
  // elided by the frontend before this pass runs, so there is nothing to
  // check).
  std::vector<std::pair<goto_programt::targett, std::vector<alloc_site_t>>>
    work;
  for (auto it = body.instructions.begin(); it != body.instructions.end(); ++it)
  {
    if (!it->is_assign())
      continue;
    std::vector<alloc_site_t> sites;
    collect_allocs(to_code_assign2t(it->code).source, expr2tc(), ns, sites);
    if (!sites.empty())
      work.emplace_back(it, std::move(sites));
  }

  const expr2tc bound = constant_int2tc(size_type2(), max_alloc_bytes);
  bool changed = false;
  for (auto &[it, sites] : work)
  {
    // Snapshot before inserting: each insert_swap splices a new ASSERT into
    // the slot `it` names and shifts the allocation one step right, so the
    // location/function are read from the allocation exactly once.
    const locationt loc = it->location;
    const irep_idt fn = it->function;
    for (const alloc_site_t &site : sites)
    {
      // `size <= K`, weakened to `guard => size <= K` when the allocation is
      // conditional so a not-taken branch cannot raise a false witness.
      expr2tc cond = lessthanequal2tc(site.byte_size, bound);
      if (!is_nil_expr(site.guard))
        cond = or2tc(not2tc(site.guard), cond);

      goto_programt new_code;
      goto_programt::targett t = new_code.add_instruction(ASSERT);
      t->guard = cond;
      t->location = loc;
      t->location.comment(std::string("excessive allocation size: ") + site.fn);
      t->location.property("excessive-allocation");
      t->function = fn;
      body.insert_swap(it, new_code.instructions.front());
    }
    changed = true;
  }

  return changed;
}
