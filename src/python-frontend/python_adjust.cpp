#include <python-frontend/python_adjust.h>

#include <irep2/irep2_utils.h>
#include <vector>

python_adjust::python_adjust(contextt &_context)
  : context(_context), ns(_context)
{
}

bool python_adjust::adjust()
{
  // Hash-table iterators are not stable across mutation, so snapshot the
  // symbol pointers first (mirrors clang_c_adjust::adjust()).
  std::vector<symbolt *> symbol_list;
  context.Foreach_operand_in_order(
    [&symbol_list](symbolt &s) { symbol_list.push_back(&s); });

  for (symbolt *symbol : symbol_list)
  {
    if (symbol->is_type)
      continue;

    expr2tc value = symbol->get_value2();
    if (is_nil_expr(value))
      continue;

    adjust_expr(value);
    symbol->set_value(value);
  }

  return false;
}

void python_adjust::adjust_expr(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  // B.0: structural no-op — recurse into every operand and change nothing.
  // The recursion shape is the durable part; later phases resolve
  // member2t/index2t sources here before recursing/returning.
  expr->Foreach_operand([this](expr2tc &op) { adjust_expr(op); });
}
