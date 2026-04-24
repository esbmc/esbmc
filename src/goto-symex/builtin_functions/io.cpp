#include <cassert>
#include <goto-symex/goto_symex.h>
#include <goto-symex/printf_formatter.h>
#include <string>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/std_types.h>
#include <vector>
#include <algorithm>
#include <util/array2string.h>

void goto_symext::symex_printf(const expr2tc &lhs, expr2tc &rhs)
{
  assert(is_code_printf2t(rhs));

  expr2tc renamed_rhs = rhs;
  cur_state->rename(renamed_rhs);

  code_printf2t &new_rhs = to_code_printf2t(renamed_rhs);

  if (new_rhs.bs_name.empty())
  {
    log_error("No base_name for code_printf2t");
    return;
  }

  const std::string &base_name = new_rhs.bs_name;

  // get the format string base on the bs_name
  irep_idt fmt;
  size_t idx;
  if (base_name == "printf")
  {
    // 1. printf: 1st argument
    assert(new_rhs.operands.size() >= 1 && "Wrong printf signature");
    const expr2tc &base_expr = get_base_object(new_rhs.operands[0]);
    if (is_constant_string2t(base_expr))
    {
      fmt = to_constant_string2t(base_expr).value;
      idx = 1;
    }
    else
    {
      // e.g.
      // int x = 1;
      // printf(x); // output ""
      fmt = "";
      idx = 0;
    }
  }
  else if (
    base_name == "fprintf" || base_name == "dprintf" ||
    base_name == "sprintf" || base_name == "vfprintf")
  {
    // 2.fprintf, sprintf, dprintf: 2nd argument
    assert(
      new_rhs.operands.size() >= 2 &&
      "Wrong fprintf/sprintf/dprintf/vfprintf signature");
    const expr2tc &base_expr = get_base_object(new_rhs.operands[1]);
    if (is_constant_string2t(base_expr))
    {
      fmt = to_constant_string2t(base_expr).value;
      idx = 2;
    }
    else
    {
      fmt = "";
      idx = 1;
    }
  }
  else if (base_name == "snprintf")
  {
    // 3. snprintf: 3rd argument
    assert(new_rhs.operands.size() >= 3 && "Wrong snprintf signature");
    const expr2tc &base_expr = get_base_object(new_rhs.operands[2]);
    if (is_constant_string2t(base_expr))
    {
      fmt = to_constant_string2t(base_expr).value;
      idx = 3;
    }
    else
    {
      fmt = "";
      idx = 2;
    }
  }
  else
    abort();

  // Check format specifiers against original operands before renaming/conversion
  if (options.get_bool_option("printf-check"))
  {
    const code_printf2t &original_rhs = to_code_printf2t(rhs);
    std::string format_str = fmt.as_string();
    size_t arg_idx = 0;

    for (size_t i = 0; i < format_str.length(); i++)
    {
      if (format_str[i] == '%')
      {
        if (i + 1 < format_str.length() && format_str[i + 1] == '%')
        {
          i++; // Skip %%
          continue;
        }

        // Skip flags, width, precision
        i++;
        while (i < format_str.length() &&
               (format_str[i] == '-' || format_str[i] == '+' ||
                format_str[i] == ' ' || format_str[i] == '#' ||
                format_str[i] == '0'))
          i++;
        while (i < format_str.length() && isdigit(format_str[i]))
          i++;
        if (i < format_str.length() && format_str[i] == '.')
        {
          i++;
          while (i < format_str.length() && isdigit(format_str[i]))
            i++;
        }

        // Skip length modifiers
        while (i < format_str.length() &&
               (format_str[i] == 'h' || format_str[i] == 'l' ||
                format_str[i] == 'L' || format_str[i] == 'z' ||
                format_str[i] == 'j' || format_str[i] == 't'))
          i++;

        // Check conversion specifier against original operands
        if (i < format_str.length())
        {
          char spec = format_str[i];
          size_t actual_arg_idx = idx + arg_idx;

          // Check if we have enough arguments (skip %n and %*)
          if (spec != 'n' && spec != '*')
          {
            if (actual_arg_idx >= original_rhs.operands.size())
            {
              claim(
                gen_false_expr(),
                "printf has more format specifiers than arguments");
            }
            else
            {
              const expr2tc &arg = original_rhs.operands[actual_arg_idx];

              if (arg)
              {
                if (spec == 's' || spec == 'p')
                {
                  // %s and %p require pointer types
                  if (!is_pointer_type(arg->type))
                  {
                    claim(
                      gen_false_expr(),
                      spec == 's'
                        ? "printf format specifier %s requires pointer argument"
                        : "printf format specifier %p requires pointer "
                          "argument");
                  }
                }
              }
            }
            arg_idx++;
          }
        }
      }
    }
  }

  // Only perform dereference checks if printf-check is enabled
  if (options.get_bool_option("printf-check"))
  {
    // Dereference check all pointer arguments after the format string
    for (size_t i = idx; i < new_rhs.operands.size(); i++)
    {
      expr2tc &arg = new_rhs.operands[i];

      if (!arg)
        continue;

      if (!is_pointer_type(arg->type))
        continue;

      if (cur_state->guard.is_false())
        continue;

      // Check the entire expression tree for L2 symbols, not just top-level
      bool has_l2_symbols = false;
      arg->foreach_operand([&has_l2_symbols](const expr2tc &e) {
        if (is_symbol2t(e))
        {
          const symbol2t &sym = to_symbol2t(e);
          if (
            sym.rlevel == symbol2t::renaming_level::level2 ||
            sym.rlevel == symbol2t::renaming_level::level2_global)
          {
            has_l2_symbols = true;
          }
        }
      });

      if (has_l2_symbols)
        continue;

      type2tc subtype = to_pointer_type(arg->type).subtype;

      if (is_empty_type(subtype) || is_nil_type(subtype))
        continue;

      expr2tc deref_expr = dereference2tc(subtype, arg);
      dereference(deref_expr, dereferencet::READ);
    }
  }

  // Now we pop the format
  for (size_t i = 0; i < idx; i++)
    new_rhs.operands.erase(new_rhs.operands.begin());

  std::list<expr2tc> args;
  new_rhs.foreach_operand([this, &args](const expr2tc &e) {
    expr2tc tmp = e;
    do_simplify(tmp);
    args.push_back(tmp);
  });

  if (!is_nil_expr(lhs))
  {
    // get the return value from code_printf2tc
    // 1. covert code_printf2tc back to sideeffect2tc
    exprt rhs_expr = migrate_expr_back(rhs);
    exprt printf_code("sideeffect", migrate_type_back(lhs->type));

    printf_code.statement("printf2");
    printf_code.operands() = rhs_expr.operands();
    printf_code.location() = rhs_expr.location();

    migrate_expr(printf_code, rhs);

    // 2 check if it is a char array. if so, convert it to a string
    // this is due to printf_formatter does not handle the char array.
    for (auto &arg : args)
    {
      const expr2tc &base_expr = get_base_object(arg);
      if (!is_constant_string2t(base_expr) && is_array_type(base_expr))
      {
        if (!is_symbol2t(base_expr))
        {
          log_debug(
            "printf",
            "symex_printf: array base is not a symbol (id={}), skipping "
            "array->string conversion",
            get_expr_id(base_expr));
          continue;
        }
        const symbolt &s = *ns.lookup(to_symbol2t(base_expr).thename);
        exprt dest;
        if (array2string(s, dest))
          continue;
        migrate_expr(dest, arg);
      }
    }

    // 3 get the number of characters output (return value)
    printf_formattert printf_formatter;
    printf_formatter(fmt.as_string(), args);
    printf_formatter.as_string(); // populate min_outlen / max_outlen

    // 4. do assign: constant when fully determined, bounded nondet otherwise
    if (printf_formatter.min_outlen == printf_formatter.max_outlen)
    {
      symex_assign(code_assign2tc(
        lhs,
        constant_int2tc(int_type2(), BigInt(printf_formatter.max_outlen))));
    }
    else
    {
      expr2tc nondet = sideeffect2tc(
        int_type2(),
        expr2tc(),
        expr2tc(),
        std::vector<expr2tc>(),
        type2tc(),
        sideeffect2t::nondet);
      replace_nondet(nondet);
      expr2tc lo =
        constant_int2tc(int_type2(), BigInt(printf_formatter.min_outlen));
      expr2tc hi =
        constant_int2tc(int_type2(), BigInt(printf_formatter.max_outlen));
      assume(
        and2tc(greaterthanequal2tc(nondet, lo), lessthanequal2tc(nondet, hi)));
      symex_assign(code_assign2tc(lhs, nondet));
    }
  }

  target->output(
    cur_state->guard.as_expr(), cur_state->source, fmt.as_string(), args);
}

void goto_symext::symex_input(const code_function_call2t &func_call)
{
  assert(is_symbol2t(func_call.function));

  unsigned fmt_idx;
  const irep_idt func_name = to_symbol2t(func_call.function).thename;

  if (func_name == "c:@F@scanf")
  {
    assert(func_call.operands.size() >= 2 && "Wrong scanf signature");
    fmt_idx = 0;
  }
  else if (func_name == "c:@F@fscanf" || func_name == "c:@F@sscanf")
  {
    assert(func_call.operands.size() >= 3 && "Wrong fscanf/sscanf signature");
    fmt_idx = 1;
  }
  else
    abort();

  cur_state->source.pc--;

  // Get the format string and count actual format specifiers
  expr2tc fmt_operand = func_call.operands[fmt_idx];
  cur_state->rename(fmt_operand);

  unsigned actual_format_count = 0;

  // Try to get the format string value to count specifiers
  const expr2tc &base_expr = get_base_object(fmt_operand);
  if (is_constant_string2t(base_expr))
  {
    std::string format_str = to_constant_string2t(base_expr).value.as_string();

    // Count format specifiers in the string
    // This is a simplified parser - handles %d, %s, %c, %f, etc.
    // but not complex cases like %*d (ignored), %10d (width), etc.
    for (size_t i = 0; i < format_str.length(); ++i)
    {
      if (format_str[i] == '%')
      {
        if (i + 1 < format_str.length())
        {
          if (format_str[i + 1] == '%')
          {
            // %% is an escaped %, not a format specifier
            ++i; // skip the second %
            continue;
          }
          else
          {
            // Skip any flags, width, precision specifiers
            ++i;
            while (i < format_str.length() &&
                   (format_str[i] == '-' || format_str[i] == '+' ||
                    format_str[i] == ' ' || format_str[i] == '#' ||
                    format_str[i] == '0'))
              ++i;

            // Skip width
            while (i < format_str.length() && isdigit(format_str[i]))
              ++i;

            // Skip precision
            if (i < format_str.length() && format_str[i] == '.')
            {
              ++i;
              while (i < format_str.length() && isdigit(format_str[i]))
                ++i;
            }

            // Skip length modifiers (h, l, ll, etc.)
            while (i < format_str.length() &&
                   (format_str[i] == 'h' || format_str[i] == 'l' ||
                    format_str[i] == 'L' || format_str[i] == 'z' ||
                    format_str[i] == 'j' || format_str[i] == 't'))
              ++i;
            // Check for actual conversion specifier
            if (i < format_str.length())
            {
              char spec = format_str[i];
              if (
                spec == 'd' || spec == 'i' || spec == 'o' || spec == 'u' ||
                spec == 'x' || spec == 'X' || spec == 'f' || spec == 'F' ||
                spec == 'e' || spec == 'E' || spec == 'g' || spec == 'G' ||
                spec == 'a' || spec == 'A' || spec == 'c' || spec == 's' ||
                spec == 'p' || spec == 'n')
              {
                // %n still needs a parameter even though it doesn't consume input
                actual_format_count++;
              }
            }
          }
        }
      }
    }
  }
  else
  {
    // If we can't determine the format string statically,
    // fall back to processing all provided arguments
    actual_format_count = func_call.operands.size() - (fmt_idx + 1);
  }

  // Limit to available arguments
  unsigned available_args = func_call.operands.size() - (fmt_idx + 1);
  unsigned args_to_process = std::min(actual_format_count, available_args);

  if (func_call.ret)
    symex_assign(code_assign2tc(
      func_call.ret, constant_int2tc(int_type2(), BigInt(args_to_process))));

  // TODO: fill / cut off the inputs stream based on the length limits.

  for (unsigned i = 0; i < args_to_process; i++)
  {
    expr2tc operand = func_call.operands[fmt_idx + 1 + i];
    internal_deref_items.clear();
    expr2tc deref = dereference2tc(get_empty_type(), operand);
    dereference(deref, dereferencet::INTERNAL);

    for (const auto &item : internal_deref_items)
    {
      assert(is_symbol2t(item.object) && "This only works for variables");

      auto type = item.object->type;
      expr2tc val = sideeffect2tc(
        type,
        expr2tc(),
        expr2tc(),
        std::vector<expr2tc>(),
        type2tc(),
        sideeffect2t::nondet);

      symex_assign(code_assign2tc(item.object, val), false, cur_state->guard);
    }
  }

  cur_state->source.pc++;
}
