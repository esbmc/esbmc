#include <python-frontend/python_adjust.h>

python_adjust::python_adjust(contextt &_context)
  : context(_context), ns(_context)
{
}

bool python_adjust::adjust()
{
  // warning! hash-table iterators are not stable — snapshot first, exactly as
  // clang_c_adjust::adjust() does.
  symbol_listt symbol_list;
  context.Foreach_operand_in_order(
    [&symbol_list](symbolt &s) { symbol_list.push_back(&s); });

  Forall_symbol_list(it, symbol_list)
  {
    symbolt &symbol = **it;
    if (symbol.is_type)
      continue;

    adjust_expr(symbol.get_value2());
  }

  return false;
}

void python_adjust::adjust_expr(const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  // B.0: no transformation. Descend into every operand so that B.1
  // (member/index source following) extends a complete traversal.
  expr->foreach_operand([this](const expr2tc &e) { adjust_expr(e); });
}
