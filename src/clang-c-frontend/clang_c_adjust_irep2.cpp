#include <clang-c-frontend/clang_c_adjust_irep2.h>
#include <util/irep/migrate.h>
#include <util/lang/c_typecast.h>
#include <util/lang/c_types.h>
#include <irep2/irep2_utils.h>
#include <util/symtab/namespace.h>
#include <utility>

bool clang_c_adjust_irep2::adjust()
{
  // migrate_expr resolves a symbol's type through this thread-local namespace.
  // typecheck builds into its own context and c_link merges into the global one
  // afterwards, so the namespace language_ui installed does not contain these
  // symbols yet: every lookup would miss and fall through to
  // sym_name_to_symbol's renaming parser, which warns once per symbol. Point it
  // at the context being built, as dereferencet does for the same reason.
  namespacet ns(context);
  const namespacet *old_ns = std::exchange(migrate_namespace_lookup, &ns);

  // Hash-table iterators are not stable across mutation, so snapshot the
  // symbol pointers first (mirrors clang_c_adjust::adjust()).
  std::vector<symbolt *> symbol_list;
  context.Foreach_operand_in_order(
    [&symbol_list](symbolt &s) { symbol_list.push_back(&s); });

  for (symbolt *s : symbol_list)
  {
    if (!s->is_type && s->get_value().is_not_nil())
    {
      const expr2tc before = s->get_value2();
      expr2tc value = before;
      adjust_expr(value);
      // Only write back a value this pass actually changed, so symbols it does
      // not touch never make the round trip (python_adjust takes the same care,
      // for the bitfield and alignment losses migrate_type cannot carry).
      if (value != before)
        s->set_value(value);
    }
  }

  migrate_namespace_lookup = old_ns;
  return false;
}

void clang_c_adjust_irep2::adjust_expr(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  expr->Foreach_operand([this](expr2tc &op) { adjust_expr(op); });

  if (is_index2t(expr))
    adjust_index(expr);
}

void clang_c_adjust_irep2::adjust_index(expr2tc &expr)
{
  const index2t &idx = to_index2t(expr);
  expr2tc array_expr = idx.source_value;
  expr2tc index_expr = idx.index;

  // The operands may be the other way round: `i[a]` is legal C.
  if (const type2tc a = ns.follow(array_expr->type),
      i = ns.follow(index_expr->type);
      !is_array_type(a) && !is_pointer_type(a) &&
      (is_array_type(i) || is_pointer_type(i)))
    std::swap(array_expr, index_expr);

  // migrate_type(index_type()), not index_type2(): despite the name they are
  // different types -- index_type() is signed_size_type(), index_type2() is
  // get_int_type(config.ansi_c.address_width) -- so the latter leaves a
  // non-constant subscript uncast where the legacy arm casts it.
  c_implicit_typecast(index_expr, migrate_type(index_type()), ns);

  const type2tc final_array_type = ns.follow(array_expr->type);

  // p[i] is syntactic sugar for *(p+i).
  if (is_pointer_type(final_array_type))
  {
    expr = dereference2tc(
      expr->type, add2tc(array_expr->type, array_expr, index_expr));
    return;
  }

  if (!is_array_type(final_array_type) && !is_vector_type(final_array_type))
  {
    // The base is neither array, vector nor pointer -- typically a struct that
    // appears in array context because two TUs declared the same external
    // symbol with conflicting types. Rewrite `base[i]` as `*((T*)&base + i)`.
    const type2tc elem_type = expr->type;
    const type2tc ptr_type = pointer_type2tc(elem_type);

    expr2tc addr_of =
      address_of2tc(pointer_type2tc(array_expr->type), array_expr);
    c_implicit_typecast(addr_of, ptr_type, ns);

    expr = dereference2tc(elem_type, add2tc(ptr_type, addr_of, index_expr));
    return;
  }

  expr = index2tc(expr->type, array_expr, index_expr);
}
