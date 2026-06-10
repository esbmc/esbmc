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
    // Replace RACE_CHECK(&x) with __ESBMC_races_flag[key(&x)].
    const races_check2t &obj = to_races_check2t(expr);

    expr2tc flag;
    migrate_expr(symbol_expr(*ns.lookup("c:@F@__ESBMC_races_flag")), flag);

    // Index __ESBMC_races_flag by a word-sized key that packs the access's
    // pointer object into the high half of the word and its (masked) byte offset
    // into the low half: (object << word_size/2) | (offset & (2^(word_size/2)-1)).
    // The races_flag array domain is the machine word, so the key must stay
    // word-sized; placing the object in the high half and masking the offset to
    // the low half guarantees an offset can never reach another object's bits.
    // Distinct objects therefore never alias as long as the object number and
    // the in-object offset each fit in word_size/2 bits -- which they do for the
    // accesses these checks instrument. The previous encoding flattened the two
    // into `object * 1000 + offset`, where any offset of 1000 or more (e.g. a
    // write to arr[i] with a large i) spilled into the next object's band and
    // fabricated a data race. See issue #5137.
    //
    // pointer_object/pointer_offset lower to projections of the pointer tuple,
    // whose fields are address_width wide regardless of the type carried here;
    // cast both to key_type so the bitwise ops below see matching widths even on
    // data models where address_width != word_size (e.g. LP32).
    unsigned int half = config.ansi_c.word_size / 2;
    type2tc key_type = get_uint_type(config.ansi_c.word_size);

    expr2tc object =
      typecast2tc(key_type, pointer_object2tc(ptraddr_type2(), obj.value));
    expr2tc offset =
      typecast2tc(key_type, pointer_offset2tc(ptraddr_type2(), obj.value));
    expr2tc index = bitor2tc(
      key_type,
      shl2tc(key_type, object, constant_int2tc(key_type, BigInt(half))),
      bitand2tc(
        key_type, offset, constant_int2tc(key_type, BigInt::power2m1(half))));

    expr = index2tc(get_bool_type(), flag, index);
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
    if (sym && sym->get_type().cmt_volatile())
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
