#include <clang-c-frontend/clang_c_adjust_irep2.h>

bool clang_c_adjust_irep2::adjust()
{
  // Hash-table iterators are not stable across mutation, so snapshot the
  // symbol pointers first (mirrors clang_c_adjust::adjust()).
  std::vector<symbolt *> symbol_list;
  context.Foreach_operand_in_order(
    [&symbol_list](symbolt &s) { symbol_list.push_back(&s); });

  for (symbolt *s : symbol_list)
  {
    if (!s->is_type && s->get_value().is_not_nil())
    {
      expr2tc value = s->get_value2();
      adjust_expr(value);
    }
  }

  return false;
}

void clang_c_adjust_irep2::adjust_expr(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  expr->Foreach_operand([this](expr2tc &op) { adjust_expr(op); });
}
