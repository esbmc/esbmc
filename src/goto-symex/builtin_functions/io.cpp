#include <cassert>
#include <climits>
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
#include <functional>
#include <util/array2string.h>

bool goto_symext::recover_va_list_args(
  const code_printf2t &call,
  size_t fmt_idx,
  std::list<expr2tc> &out)
{
  switch (call.kind)
  {
  case printf_kindt::VPRINTF:
  case printf_kindt::VFPRINTF:
  case printf_kindt::VSPRINTF:
  case printf_kindt::VSNPRINTF:
  case printf_kindt::VASPRINTF:
    break;
  default:
    return false;
  }

  // Every v* signature takes the va_list as its last operand, right after
  // the format string.
  if (call.operands.size() != fmt_idx + 2)
    return false;

  // The frontends model a va_list as an opaque token: under the clang
  // frontend va_start assigns nothing, and va_arg reads the frame-materialised
  // `<fn>::va_arg<N>` symbols through the frame cursor. The only state that
  // identifies which arguments a va_list denotes is therefore the frame
  // itself, so recovery is restricted to the case where that identification
  // is exact:
  //   1. the v* call sits in the variadic function's own frame (its va_args
  //      were materialised at this activation's call site);
  //   2. no va_arg has been consumed yet (once the cursor moved, a va_start
  //      rewind -- invisible to symex -- is indistinguishable from further
  //      consumption, so the specifier-to-argument alignment is unknown);
  //   3. the va_list operand is one of this function's own locals, not a
  //      formal parameter (a parameter could carry a va_list from a *further*
  //      variadic ancestor, whose arguments these are not).
  // Anything else declines; the caller keeps the sound unbounded fallback.
  const goto_symex_statet::framet &frame = cur_state->top();
  if (frame.va_index == UINT_MAX || frame.va_cursor != frame.va_index)
    return false;

  expr2tc ap = call.operands[fmt_idx + 1];
  while (true)
  {
    if (is_typecast2t(ap))
      ap = to_typecast2t(ap).from;
    else if (is_address_of2t(ap))
      ap = to_address_of2t(ap).ptr_obj;
    else if (is_index2t(ap))
      ap = to_index2t(ap).source_value;
    else
      break;
  }
  if (!is_symbol2t(ap))
    return false;
  const irep_idt &ap_id = to_symbol2t(ap).thename;

  // Recovery is limited to targets where va_list is a plain pointer (e.g.
  // AArch64 SysV/Darwin): there every va_list copy is a visible ASSIGN --
  // va_copy included, since goto conversion lowers it to a real assignment
  // for pointer va_lists (builtin_functions.cpp) -- so the freshness scan
  // below is exhaustive. Where va_list is a struct array (x86_64 SysV),
  // va_copy stays erased at goto conversion (assigning arrays breaks the
  // pointer analysis), a copied foreign va_list would be invisible, and
  // recovery must stay off until va_copy leaves a symex-visible marker
  // there too.
  if (!is_pointer_type(ap->type))
    return false;

  // Owning-function check: a local of function F is mangled
  // `<file>@<offset>@F@<F>@<name>` while F itself is `<prefix>@F@<F>`. Only
  // F's own locals/parameters (and globals, which carry no `@F@`) can appear
  // syntactically as an operand in F's body, so containing `@F@<F>@` pins the
  // symbol to this frame's function.
  const std::string fid = id2string(frame.function_identifier);
  const size_t at = fid.rfind("@F@");
  if (at == std::string::npos)
    return false;
  if (id2string(ap_id).find(fid.substr(at) + "@") == std::string::npos)
    return false;

  // Exclude formal parameters (condition 3).
  const symbolt *fsym = ns.lookup(frame.function_identifier);
  if (fsym == nullptr || !fsym->get_type().is_code())
    return false;
  for (const auto &param : to_code_type(fsym->get_type()).arguments())
    if (param.get_identifier() == ap_id)
      return false;

  // Freshness scan (condition 3 continued): being a non-parameter local is
  // not enough -- a foreign va_list can be laundered into one (va_copy from
  // a parameter or a global, plain assignment on pointer-va_list targets, or
  // memcpy), after which it denotes some OTHER activation's arguments. All
  // such channels are visible in the function body on pointer-va_list
  // targets: decline unless the operand's only assignment is its
  // declaration-time nondet and its address is never taken.
  auto fn_it = goto_functions.function_map.find(frame.function_identifier);
  if (
    fn_it == goto_functions.function_map.end() || !fn_it->second.body_available)
    return false;
  for (const auto &insn : fn_it->second.body.instructions)
  {
    if (is_nil_expr(insn.code))
      continue;
    if (is_code_assign2t(insn.code))
    {
      const code_assign2t &assign = to_code_assign2t(insn.code);
      expr2tc tgt = assign.target;
      while (is_typecast2t(tgt))
        tgt = to_typecast2t(tgt).from;
      if (
        is_symbol2t(tgt) && to_symbol2t(tgt).thename == ap_id &&
        !(is_sideeffect2t(assign.source) &&
          to_sideeffect2t(assign.source).kind ==
            sideeffect2t::allockind::nondet))
        return false;
    }
    // Address taken anywhere (e.g. memcpy(&ap, ...)): treat as laundered.
    bool addr_taken = false;
    std::function<void(const expr2tc &)> scan = [&](const expr2tc &e)
    {
      if (is_nil_expr(e) || addr_taken)
        return;
      if (is_address_of2t(e))
      {
        expr2tc obj = to_address_of2t(e).ptr_obj;
        while (is_typecast2t(obj) || is_index2t(obj))
          obj = is_typecast2t(obj) ? to_typecast2t(obj).from
                                   : to_index2t(obj).source_value;
        if (is_symbol2t(obj) && to_symbol2t(obj).thename == ap_id)
        {
          addr_taken = true;
          return;
        }
      }
      e->foreach_operand(scan);
    };
    scan(insn.code);
    if (addr_taken)
      return false;
  }

  // Collect this activation's va_arg values, L2-renamed so constant
  // propagation exposes string-literal arguments to the formatter. Probing
  // stops at the first missing symbol: later activations have not
  // materialised yet, so the run ends exactly at this activation's argument
  // count (a nested completed activation of the same function can extend it,
  // but those extra values are only ever consumed by a format string with
  // more specifiers than passed arguments -- undefined behaviour).
  const std::string va_prefix = fid + "::va_arg";
  for (unsigned k = frame.va_index; k < frame.va_index + 256; k++)
  {
    const symbolt *s = new_context.find_symbol(va_prefix + std::to_string(k));
    if (s == nullptr)
      break;
    expr2tc arg = symbol2tc(migrate_symbol_type(*s), s->id);
    cur_state->rename(arg);
    do_simplify(arg);
    out.push_back(arg);
  }
  return true;
}

void goto_symext::symex_printf(const expr2tc &lhs, expr2tc &rhs)
{
  assert(is_code_printf2t(rhs));

  expr2tc renamed_rhs = rhs;
  cur_state->rename(renamed_rhs);

  code_printf2t &new_rhs = to_code_printf2t(renamed_rhs);

  // Position of the format-string argument in `operands`, indexed by
  // printf_kindt: printf takes it as arg 0, fprintf/dprintf/sprintf/
  // vfprintf as arg 1, snprintf as arg 2.  Default-init to silence
  // GCC's -Wmaybe-uninitialized (it can't see that the switch is total
  // over the enum class).
  size_t fmt_idx = 0;
  switch (new_rhs.kind)
  {
  case printf_kindt::PRINTF:
  case printf_kindt::VPRINTF:
    fmt_idx = 0;
    break;
  case printf_kindt::FPRINTF:
  case printf_kindt::DPRINTF:
  case printf_kindt::SPRINTF:
  case printf_kindt::VFPRINTF:
  case printf_kindt::VSPRINTF:
  case printf_kindt::ASPRINTF:
  case printf_kindt::VASPRINTF:
    fmt_idx = 1;
    break;
  case printf_kindt::SNPRINTF:
  case printf_kindt::VSNPRINTF:
    fmt_idx = 2;
    break;
  }
  assert(new_rhs.operands.size() > fmt_idx && "Wrong printf-family signature");

  irep_idt fmt;
  size_t idx;
  const expr2tc &base_expr = get_base_object(new_rhs.operands[fmt_idx]);
  const bool format_is_constant = is_constant_string2t(base_expr);
  if (format_is_constant)
  {
    fmt = to_constant_string2t(base_expr).value;
    idx = fmt_idx + 1;
  }
  else
  {
    // e.g.   int x = 1; printf(x); // output ""
    // The format string is not known at compile time, so the output length
    // cannot be bounded; the return value is handled as unbounded below.
    fmt = "";
    idx = fmt_idx;
  }

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

  // For asprintf/vasprintf, save the char **strp argument (operand[0])
  // before erasing the leading arguments so we can later model the side-effect
  // on *strp. Without this, *strp keeps its uninitialized nondet value, causing
  // value-set aliasing and spurious memsafety false alarms (GitHub #5139,
  // #5140, #5141).
  const bool is_allocating = new_rhs.kind == printf_kindt::ASPRINTF ||
                             new_rhs.kind == printf_kindt::VASPRINTF;
  expr2tc strp;
  if (is_allocating && !new_rhs.operands.empty())
    strp = new_rhs.operands[0];

  // va_list %s recovery (G-C, GitHub #5012): when the va_list-to-argument
  // mapping is provably exact, substitute the recovered arguments for the
  // opaque va_list below, so the formatter can bound -- or pin -- the output
  // length exactly as it does for the direct printf-family forms. Uses the
  // original (unrenamed) rhs: the operand's L0 symbol identifies whose
  // va_list this is.
  std::list<expr2tc> recovered_args;
  const bool va_recovered =
    !is_nil_expr(lhs) && format_is_constant &&
    recover_va_list_args(to_code_printf2t(rhs), fmt_idx, recovered_args);

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

    // With a successful va_list recovery the single va_list operand is
    // replaced by the actual variadic arguments, restoring the direct-call
    // argument shape the formatter expects.
    if (va_recovered)
      args = recovered_args;

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
        // Only fold to a string constant when the array has compile-time
        // content. A nondet / runtime-filled array has no static initializer;
        // leaving it as an array lets the formatter apply a sound object-size
        // length bound (strlen <= size-1) instead of under-approximating the
        // %s contribution to "" (length 0).
        const exprt &val = s.get_value();
        if (val.is_nil() || val.operands().empty())
          continue;
        exprt dest;
        if (array2string(s, dest))
          continue;
        migrate_expr(dest, arg);
      }
    }

    // 3 get the number of characters output (return value)
    printf_formattert printf_formatter;
    // For the v* variants the actual arguments are hidden behind a va_list, so
    // the operand at a %s position is the va_list pointer, not the string. Mark
    // the operands unreliable so the formatter does not derive an (unsound)
    // object-size bound from them; %s stays unbounded until va_list recovery.
    // Exhaustive over printf_kindt (no default) so a future va_list kind must
    // be classified here rather than silently inheriting the reliable default.
    switch (new_rhs.kind)
    {
    case printf_kindt::VPRINTF:
    case printf_kindt::VFPRINTF:
    case printf_kindt::VSPRINTF:
    case printf_kindt::VSNPRINTF:
    case printf_kindt::VASPRINTF:
      printf_formatter.args_reliable = false;
      break;
    case printf_kindt::PRINTF:
    case printf_kindt::FPRINTF:
    case printf_kindt::DPRINTF:
    case printf_kindt::SPRINTF:
    case printf_kindt::SNPRINTF:
    case printf_kindt::ASPRINTF:
      printf_formatter.args_reliable = true;
      break;
    }
    // A successful recovery replaced the va_list with the actual arguments,
    // so they are exactly as reliable as a direct call's.
    if (va_recovered)
      printf_formatter.args_reliable = true;
    printf_formatter(fmt.as_string(), args);
    printf_formatter.as_string(); // populate min_outlen / max_outlen

    // 4. do assign. The return value is the number of characters that would be
    //    written. We can pin it to an exact constant only when the format
    //    string is a compile-time constant AND every conversion was soundly
    //    bounded; we can bound it to [min,max] when those hold but some
    //    argument is nondet; otherwise there is no sound upper bound.
    //
    //    For asprintf/vasprintf specifically, the return value is -1 on
    //    allocation failure and the output length on success. When the format
    //    is not soundly bounded (e.g. a %s with a non-literal argument), we
    //    cap the return at INT_MAX/2 (host int is 32-bit on all supported
    //    targets). This prevents spurious signed-overflow alarms on subsequent
    //    arithmetic such as `applet_len + used` in busybox (GitHub #5144)
    //    while still catching real overflows: any genuine overflow in such
    //    arithmetic requires used to exceed INT_MAX/2, which in turn demands a
    //    >1 GB formatted string — infeasible in practice. Tightening further
    //    would risk masking real overflows; see also GitHub #4976-#4979.
    const bool sound_bound = format_is_constant && printf_formatter.bounded;
    const bool is_allocating = new_rhs.kind == printf_kindt::ASPRINTF ||
                               new_rhs.kind == printf_kindt::VASPRINTF;
    if (
      sound_bound && printf_formatter.min_outlen == printf_formatter.max_outlen)
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
        sideeffect2t::allockind::nondet);
      replace_nondet(nondet);
      if (sound_bound)
      {
        expr2tc lo =
          constant_int2tc(int_type2(), BigInt(printf_formatter.min_outlen));
        expr2tc hi =
          constant_int2tc(int_type2(), BigInt(printf_formatter.max_outlen));
        assume(and2tc(
          greaterthanequal2tc(nondet, lo), lessthanequal2tc(nondet, hi)));
      }
      else if (is_allocating)
      {
        // -1 on allocation failure; cap success side at INT_MAX/2.
        expr2tc lo = constant_int2tc(int_type2(), BigInt(-1));
        expr2tc hi = constant_int2tc(int_type2(), BigInt(INT_MAX / 2));
        assume(and2tc(
          greaterthanequal2tc(nondet, lo), lessthanequal2tc(nondet, hi)));
      }
      else
        assume(
          greaterthanequal2tc(nondet, constant_int2tc(int_type2(), BigInt(0))));
      symex_assign(code_assign2tc(lhs, nondet));
    }
  }

  // Model *strp for asprintf/vasprintf: assign a fresh tracked heap allocation.
  // The buffer size is modelled as 1 byte; exact sizing requires va_list
  // recovery (G-C, not yet implemented). With --no-bounds-check this is
  // sufficient to eliminate the false alarms while exact size analysis is
  // deferred. Users running with --bounds-check should be aware of this
  // limitation.
  if (is_allocating && !is_nil_expr(strp) && is_pointer_type(strp->type))
  {
    // Derive char * from strp's declared type (char **) so the dereference
    // width matches what the value-set analysis and SMT encoding expect.
    type2tc char_ptr_type = to_pointer_type(strp->type).subtype;
    expr2tc deref_strp = dereference2tc(char_ptr_type, strp);
    expr2tc malloc_se = sideeffect2tc(
      char_ptr_type,
      expr2tc(),
      constant_int2tc(size_type2(), BigInt(1)),
      std::vector<expr2tc>(),
      char_type2(),
      sideeffect2t::allockind::malloc);
    symex_assign(code_assign2tc(deref_strp, malloc_se));
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
        sideeffect2t::allockind::nondet);

      symex_assign(code_assign2tc(item.object, val), false, cur_state->guard);
    }
  }

  cur_state->source.pc++;
}
