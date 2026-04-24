#include <goto-symex/goto_symex.h>
#include <string>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <irep2/irep2.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/std_types.h>

void goto_symext::replace_races_check(expr2tc &expr)
{
  if (!options.get_bool_option("data-races-check"))
    return;

  // replace RACE_CHECK(&x) with __ESBMC_races_flag[&x]
  // recursion is needed for this case: !RACE_CHECK(&x)
  expr->Foreach_operand([this](expr2tc &e) {
    if (!is_nil_expr(e))
      replace_races_check(e);
  });

  if (is_races_check2t(expr))
  {
    // replace with __ESBMC_races_flag[index]
    const races_check2t &obj = to_races_check2t(expr);

    expr2tc flag;
    migrate_expr(symbol_expr(*ns.lookup("c:@F@__ESBMC_races_flag")), flag);

    expr2tc max_offset =
      constant_int2tc(get_uint_type(config.ansi_c.address_width), 1000);
    // The reason for not using address directly is that address
    // is modeled as an nondet value, which depends on the address space constraints.
    // VCC becomes complex and inefficient in this case.

    // The current method is similar to a two-dimensional array: array[obj][offset]
    // But we flatten it out: obj * MAX_VALUE + offset
    // In theory, this should create a unique index for variables.
    // We need to think carefully about the value of MAX_VALUE
    // XL: Should we let the user choose this value?
    expr2tc mul = mul2tc(
      size_type2(), pointer_object2tc(pointer_type2(), obj.value), max_offset);
    expr2tc add = add2tc(
      size_type2(),
      mul,
      pointer_offset2tc(get_int_type(config.ansi_c.address_width), obj.value));

    expr2tc index_expr = index2tc(get_bool_type(), flag, add);

    expr = index_expr;
  }
}

void goto_symext::volatile_check(expr2tc &expr)
{
  if (!options.get_bool_option("volatile-check"))
    return;

  if (is_symbol2t(expr))
  {
    const symbol2t &s = to_symbol2t(expr);
    const symbolt *sym = new_context.find_symbol(s.thename);
    if (sym && sym->type.cmt_volatile())
    {
      log_debug("volatile check", "variable: {}", sym->name.as_string());
      unsigned int &nondet_count = get_nondet_counter();
      expr = symbol2tc(
        expr->type, "nondet$symex::nondet" + i2string(nondet_count++));
    }
  }
  else
  {
    expr->Foreach_operand([this](expr2tc &e) {
      if (!is_nil_expr(e))
        volatile_check(e);
    });
  }
}
