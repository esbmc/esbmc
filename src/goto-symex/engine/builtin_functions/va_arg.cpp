#include <goto-symex/engine/goto_symex.h>
#include <string>
#include <util/arith/arith_tools.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <irep2/irep2.h>
#include <util/irep/migrate.h>
#include <util/irep/std_types.h>

/* Peel array decay, casts and field/element selection off a va_list
 * expression to reach the underlying object's symbol. On e.g. x86-64
 * va_list is `struct __va_list_tag[1]`, so the expression arrives as
 * typecast(address_of(index(symbol, 0))). Returns nil if the base is
 * not a plain symbol. */
static expr2tc va_list_base(expr2tc e)
{
  while (true)
  {
    if (is_typecast2t(e))
      e = to_typecast2t(e).from;
    else if (is_address_of2t(e))
      e = to_address_of2t(e).ptr_obj;
    else if (is_index2t(e))
      e = to_index2t(e).source_value;
    else if (is_member2t(e))
      e = to_member2t(e).source_value;
    else
      break;
  }
  return is_symbol2t(e) ? e : expr2tc();
}

/* Resolve a va_list expression to the l1 identity record of the local
 * variable backing it, or nullopt when it cannot be pinned down to one
 * (base is no plain symbol, a parameter, or a static). va_list arguments
 * have been dereferenced by the time they reach us, so a va_list accessed
 * through a pointer resolves to the owning activation's l1-renamed symbol;
 * an unrenamed (l0) symbol denotes a local of the current frame and is
 * renamed here. The record is normalised to level1 so lookups match
 * regardless of the renaming level the expression arrived with. */
std::optional<renaming::level2t::name_record>
goto_symext::va_list_l1_record(const expr2tc &va_list_expr) const
{
  expr2tc base = va_list_base(va_list_expr);
  if (is_nil_expr(base))
    return std::nullopt;

  const symbolt *s = new_context.find_symbol(to_symbol2t(base).thename);
  if (s == nullptr || s->is_parameter || s->static_lifetime)
    return std::nullopt;

  if (to_symbol2t(base).rlevel == symbol2t::renaming_level::level0)
    cur_state->top().level1.get_ident_name(base);

  symbol2t sym = to_symbol2t(base);
  sym.rlevel = symbol2t::renaming_level::level1;
  return renaming::level2t::name_record(sym);
}

std::vector<renaming::level2t::name_record>
goto_symext::va_list_pointee_records(const expr2tc &va_list_expr) const
{
  std::vector<renaming::level2t::name_record> records;
  value_setst::valuest values;
  cur_state->value_set.get_value_set(va_list_expr, values);
  for (const expr2tc &v : values)
    if (is_object_descriptor2t(v))
      if (auto rec = va_list_l1_record(to_object_descriptor2t(v).object))
        records.push_back(*rec);
  return records;
}

bool goto_symext::va_list_is_started(const expr2tc &va_list_expr) const
{
  auto rec = va_list_l1_record(va_list_expr);
  return !rec || va_started.count(*rec) != 0;
}

unsigned *goto_symext::va_list_cursor(const expr2tc &va_list_expr)
{
  auto rec = va_list_l1_record(va_list_expr);
  auto it = rec ? va_started.find(*rec) : va_started.end();
  return it != va_started.end() && it->second ? &*it->second : nullptr;
}

void goto_symext::va_list_mark_started(
  const expr2tc &va_list_expr,
  bool started,
  std::optional<unsigned> cursor)
{
  auto rec = va_list_l1_record(va_list_expr);
  if (rec)
  {
    va_copied_from.erase(*rec);
    if (started)
      va_started[*rec] = cursor;
    else
      va_started.erase(*rec);
    return;
  }

  /* The destination does not resolve to one local variable, e.g. a va_copy
   * into a va_list reached through a pointer. Mark every local the pointer
   * may point to as started. Erasing on a may-point-to basis could drop a
   * genuinely started va_list, so only ever widen towards "started". */
  if (!started)
    return;

  for (const auto &obj_rec : va_list_pointee_records(va_list_expr))
    va_started[obj_rec] = cursor;
}

std::vector<renaming::level2t::name_record>
goto_symext::va_list_owner_records(const expr2tc &va_list_expr) const
{
  auto rec = va_list_l1_record(va_list_expr);
  auto records =
    rec ? std::vector{*rec} : va_list_pointee_records(va_list_expr);
  for (auto &r : records)
    if (auto it = va_copied_from.find(r); it != va_copied_from.end())
      r = it->second;
  return records;
}

goto_symex_statet::framet &
goto_symext::va_list_frame(const expr2tc &va_list_expr)
{
  for (const auto &rec : va_list_owner_records(va_list_expr))
    for (auto &frame : cur_state->call_stack)
      if (frame.local_variables.count(rec))
        return frame;
  return cur_state->top();
}

void goto_symext::va_list_copy(const expr2tc &dst, const expr2tc &src)
{
  /* A source with no cursor of its own, such as a va_list parameter, reads
   * at its frame's cursor; the copy keeps that position for itself. */
  const goto_symex_statet::framet &frame = va_list_frame(src);
  const unsigned *cursor = va_list_cursor(src);
  if (!cursor && frame.va_index != UINT_MAX)
    cursor = &frame.va_cursor;
  va_list_mark_started(
    dst,
    va_list_is_started(src),
    cursor ? std::optional{*cursor} : std::nullopt);

  const auto owners = va_list_owner_records(src);
  if (auto rec = va_list_l1_record(dst); rec && !owners.empty())
    va_copied_from[*rec] = owners.front();
}

void goto_symext::symex_va_arg(
  const expr2tc &lhs,
  const sideeffect2t &code,
  const guard2tc &guard)
{
  /* Reading through a va_list that was never initialised by va_start is
   * undefined behaviour; the positional vararg machinery below would
   * happily resolve it, silently masking the bug. Only emit the claim
   * when it is violated, so correct code gets no extra VCC. */
  if (!is_nil_expr(code.operand) && !va_list_is_started(code.operand))
    claim(
      not2tc(guard.as_expr()),
      "missing va_start: va_arg on an uninitialised va_list");

  goto_symex_statet::framet &frame = va_list_frame(code.operand);
  std::string base = id2string(frame.function_identifier) + "::va_arg";

  /* The declaring frame's cursor counts every va_arg on its arguments, which
   * symex_printf's va_list recovery reads. A va_list whose own cursor is known
   * reads from that, so va_copy, a second va_start and a second va_list each
   * read where they should. */
  unsigned cursor = frame.va_cursor++;
  if (unsigned *own = va_list_cursor(code.operand))
    cursor = (*own)++;
  irep_idt id = base + std::to_string(cursor);

  expr2tc va_rhs;

  const symbolt *s = new_context.find_symbol(id);
  if (s != nullptr)
  {
    type2tc symbol_type = migrate_symbol_type(*s);

    va_rhs = symbol2tc(symbol_type, s->id);
    frame.level1.get_ident_name(va_rhs);

    va_rhs = typecast2tc(lhs->type, va_rhs);
  }
  else
  {
    va_rhs = gen_zero(lhs->type);
  }

  symex_assign(code_assign2tc(lhs, va_rhs), true, guard);
}
