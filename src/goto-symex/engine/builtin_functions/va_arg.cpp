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

expr2tc
goto_symext::va_list_cursor_symbol(const renaming::level2t::name_record &rec)
{
  const irep_idt id = id2string(rec.base_name) + "$va_cursor";
  const type2tc type = get_uint_type(config.ansi_c.int_width);
  if (new_context.find_symbol(id) == nullptr)
  {
    symbolt symbol;
    symbol.id = id;
    symbol.name = id;
    set_symbol_type(symbol, type);
    new_context.move(symbol);
  }
  return symbol2tc(
    type, id, symbol2t::renaming_level::level1, rec.l1_num, 0, rec.t_num, 0);
}

expr2tc goto_symext::va_list_cursor(const expr2tc &va_list_expr)
{
  auto rec = va_list_l1_record(va_list_expr);
  auto it = rec ? va_started.find(*rec) : va_started.end();
  return it != va_started.end() && it->second ? va_list_cursor_symbol(*rec)
                                              : expr2tc();
}

void goto_symext::va_list_mark_started(
  const expr2tc &va_list_expr,
  bool started,
  const expr2tc &cursor)
{
  auto start = [&](const renaming::level2t::name_record &rec) {
    va_started[rec] = !is_nil_expr(cursor);
    if (!is_nil_expr(cursor))
      symex_assign(code_assign2tc(va_list_cursor_symbol(rec), cursor), true);
  };

  auto rec = va_list_l1_record(va_list_expr);
  if (rec)
  {
    va_copied_from.erase(*rec);
    if (started)
      start(*rec);
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
    start(obj_rec);
}

void goto_symext::va_list_start(const expr2tc &va_list_expr)
{
  const unsigned va_index = cur_state->top().va_index;
  va_list_mark_started(
    va_list_expr,
    true,
    va_index == UINT_MAX
      ? expr2tc()
      : constant_int2tc(
          get_uint_type(config.ansi_c.int_width), BigInt(va_index)));
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
  expr2tc cursor = va_list_cursor(src);
  if (!is_nil_expr(cursor))
    cur_state->rename(cursor);
  else if (frame.va_index != UINT_MAX)
    cursor = constant_int2tc(
      get_uint_type(config.ansi_c.int_width), BigInt(frame.va_cursor));
  va_list_mark_started(dst, va_list_is_started(src), cursor);

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
  const std::string base = id2string(frame.function_identifier) + "::va_arg";

  auto argument = [&](unsigned index) -> expr2tc {
    const symbolt *s = new_context.find_symbol(base + std::to_string(index));
    if (s == nullptr)
      return gen_zero(lhs->type);
    expr2tc arg = symbol2tc(migrate_symbol_type(*s), s->id);
    frame.level1.get_ident_name(arg);
    return typecast2tc(lhs->type, arg);
  };

  /* The declaring frame's cursor counts every va_arg on its arguments, which
   * symex_printf's va_list recovery reads. A va_list whose own cursor is known
   * reads from that, so va_copy, a second va_start and a second va_list each
   * read where they should. */
  const unsigned frame_cursor = frame.va_cursor++;
  const expr2tc own = va_list_cursor(code.operand);
  if (is_nil_expr(own))
  {
    symex_assign(code_assign2tc(lhs, argument(frame_cursor)), true, guard);
    return;
  }

  /* After a va_arg on one branch only, the cursor differs between paths and
   * is no longer a constant: select the argument it denotes. */
  expr2tc cursor = own;
  cur_state->rename(cursor);
  expr2tc va_rhs;
  if (is_constant_int2t(cursor))
    va_rhs = argument(to_constant_int2t(cursor).value.to_uint64());
  else
  {
    va_rhs = gen_zero(lhs->type);
    for (unsigned k = frame.va_index;
         frame.va_index != UINT_MAX &&
         new_context.find_symbol(base + std::to_string(k)) != nullptr;
         k++)
      va_rhs = if2tc(
        lhs->type,
        equality2tc(cursor, constant_int2tc(cursor->type, BigInt(k))),
        argument(k),
        va_rhs);
  }

  symex_assign(code_assign2tc(lhs, va_rhs), true, guard);
  symex_assign(
    code_assign2tc(
      own, add2tc(cursor->type, cursor, constant_int2tc(cursor->type, 1))),
    true,
    guard);
}
