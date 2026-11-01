#include <cassert>
#include <cctype>
#include <climits>
#include <goto-symex/engine/goto_symex.h>
#include <goto-symex/trace/printf_formatter.h>
#include <string>
#include <util/arith/arith_tools.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <irep2/irep2.h>
#include <util/message/message.h>
#include <util/irep/migrate.h>
#include <util/irep/std_types.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <util/expr/array2string.h>

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
    // Skip the va_start/va_copy markers this analysis itself relies on: they
    // reference &ap by construction, but starting (or copying into) a va_list
    // is not the address-laundering the scan below guards against. va_copy's
    // laundering, where it matters, is carried by the separate ASSIGN it emits
    // on pointer-va_list targets, which the scan still inspects.
    if (is_code_function_call2t(insn.code))
    {
      const expr2tc &fn = to_code_function_call2t(insn.code).function;
      if (is_symbol2t(fn))
      {
        const irep_idt &fn_name = to_symbol2t(fn).thename;
        if (
          fn_name == "c:@F@__builtin_va_start" ||
          fn_name == "c:@F@__builtin_va_copy")
          continue;
      }
    }
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
    std::function<void(const expr2tc &)> scan = [&](const expr2tc &e) {
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

static bool is_literal_string_start(const expr2tc &e)
{
  if (is_typecast2t(e))
    return is_literal_string_start(to_typecast2t(e).from);
  if (!is_address_of2t(e))
    return false;
  const expr2tc &obj = to_address_of2t(e).ptr_obj;
  if (is_constant_string2t(obj))
    return true;
  return is_index2t(obj) &&
         is_constant_string2t(to_index2t(obj).source_value) &&
         is_constant_int2t(to_index2t(obj).index) &&
         to_constant_int2t(to_index2t(obj).index).value.is_zero();
}

std::vector<expr2tc>
goto_symext::printf_args(const code_printf2t &call, size_t fmt_idx)
{
  // A v* call's arguments sit behind its va_list, its last operand. When they
  // cannot be recovered, a conversion that needs one is unbounded.
  if (
    call.kind == printf_kindt::VSPRINTF || call.kind == printf_kindt::VSNPRINTF)
  {
    std::list<expr2tc> recovered;
    recover_va_list_args(call, fmt_idx, recovered);
    return {recovered.begin(), recovered.end()};
  }
  return {call.operands.begin() + fmt_idx + 1, call.operands.end()};
}

bool goto_symext::format_printf_output(
  const code_printf2t &call,
  size_t fmt_idx,
  printf_formattert &formatter,
  bool &constant_args)
{
  expr2tc fmt = call.operands[fmt_idx];
  cur_state->rename(fmt);
  const expr2tc &fmt_str = get_base_object(fmt);
  if (!is_constant_string2t(fmt_str))
    return false;

  constant_args = true;
  std::list<expr2tc> args;
  for (expr2tc arg : printf_args(call, fmt_idx))
  {
    cur_state->rename(arg);
    do_simplify(arg);
    constant_args &= is_constant_int2t(arg) || is_literal_string_start(arg);
    args.push_back(arg);
  }

  formatter(to_constant_string2t(fmt_str).value.as_string(), args);
  return true;
}

// The most output characters a sprintf call stores one by one.
static const size_t max_stored_output = 1024;

// The characters an integer conversion prints in `base` for `v`, already
// converted to the conversion's type: C11 7.21.6.1p8 gives no leading zeros
// and a '-' only for a negative value.
static expr2tc digits_length(expr2tc v, unsigned base, size_t &max_len)
{
  if (v->type->get_width() < int_type2()->get_width())
    v = typecast2tc(int_type2(), v);
  const unsigned width = v->type->get_width();
  const bool is_signed = is_signedbv_type(v);
  const BigInt top = power(2, is_signed ? width - 1 : width) - 1;

  const type2tc len_type = int_type2();
  const expr2tc one = gen_one(len_type);
  const expr2tc zero = gen_zero(len_type);
  expr2tc len = one;
  max_len += is_signed ? 2 : 1;
  if (is_signed)
    len = add2tc(
      len_type,
      len,
      if2tc(len_type, lessthan2tc(v, gen_zero(v->type)), one, zero));
  for (BigInt p = base; p <= top; p *= base)
  {
    expr2tc wide = greaterthanequal2tc(v, constant_int2tc(v->type, p));
    if (is_signed)
      wide = or2tc(wide, lessthanequal2tc(v, constant_int2tc(v->type, -p)));
    len = add2tc(len_type, len, if2tc(len_type, wide, one, zero));
    max_len++;
  }
  return len;
}

// The characters the %d, %i, %u, %o, %x or %X conversion `conv` with length
// modifier `mod` prints for the renamed argument `v`, nil when it cannot be
// measured. The formatter prints `v` converted to the conversion's type,
// whatever type was passed (C11 7.21.6.1p9), so that is the value measured.
static expr2tc integer_length(
  const expr2tc &v,
  char conv,
  const std::string &mod,
  size_t &max_len)
{
  const bool is_signed = conv == 'd' || conv == 'i';
  const unsigned base = conv == 'o'                    ? 8
                        : (conv == 'x' || conv == 'X') ? 16
                                                       : 10;
  const type2tc target = printf_int_type(is_signed, mod);
  if (
    is_nil_type(target) || is_signedbv_type(target) != is_signed ||
    !is_bv_type(v))
    return expr2tc();
  return digits_length(typecast2tc(target, v), base, max_len);
}

// Read the '-' and '0' flags and the field width of the conversion at `i`,
// false when the width is past what a sprintf call stores. The flags only
// choose the padding (C11 7.21.6.1p4).
static bool parse_field_width(
  const std::string &f,
  size_t &i,
  size_t &width,
  bool &left,
  bool &zero)
{
  left = zero = false;
  while (i < f.size() && (f[i] == '-' || f[i] == '0'))
    (f[i++] == '-' ? left : zero) = true;
  width = 0;
  while (i < f.size() && std::isdigit((unsigned char)f[i]))
    if ((width = width * 10 + (f[i++] - '0')) > max_stored_output)
      return false;
  return true;
}

// A field width is the least number of characters a conversion prints
// (C11 7.21.6.1p4); `max_before` is the bound before the conversion was
// measured into `max_len`.
static expr2tc pad_to_width(
  const expr2tc &len,
  size_t width,
  size_t max_before,
  size_t &max_len)
{
  if (width == 0)
    return len;
  max_len = std::max(max_len, max_before + width);
  const expr2tc min_len = constant_int2tc(len->type, BigInt(width));
  return if2tc(len->type, lessthan2tc(len, min_len), min_len, len);
}

// The character an output field prints at a position below its length, as
// an int; empty when only the field's length is known.
using printf_charst = std::function<expr2tc(const expr2tc &)>;

static expr2tc select_char(
  const expr2tc &pos,
  const std::vector<expr2tc> &by_pos,
  const expr2tc &otherwise)
{
  expr2tc out = otherwise;
  for (size_t k = by_pos.size(); k-- > 0;)
    out = if2tc(
      out->type,
      equality2tc(pos, constant_int2tc(pos->type, BigInt(k))),
      by_pos[k],
      out);
  return out;
}

static printf_charst text_chars(const std::string &text)
{
  std::vector<expr2tc> by_pos;
  for (const char c : text)
    by_pos.push_back(constant_int2tc(int_type2(), BigInt((unsigned char)c)));
  return [by_pos](const expr2tc &pos) {
    return select_char(pos, by_pos, gen_zero(int_type2()));
  };
}

// The digit characters of the unsigned `mag` in `base`, least significant
// first, `count` of them.
static std::vector<expr2tc>
digit_chars(const expr2tc &mag, unsigned base, char conv, size_t count)
{
  const type2tc pos_type = int_type2();
  const type2tc &utype = mag->type;
  const unsigned shift = base == 8 ? 3 : base == 16 ? 4 : 0;
  std::vector<expr2tc> digits;
  BigInt place = 1;
  for (size_t k = 0; k < count; k++, place *= base)
  {
    expr2tc d = shift
                  ? bitand2tc(
                      utype,
                      lshr2tc(utype, mag, constant_int2tc(utype, k * shift)),
                      constant_int2tc(utype, BigInt(base - 1)))
                  : modulus2tc(
                      utype,
                      div2tc(utype, mag, constant_int2tc(utype, place)),
                      constant_int2tc(utype, BigInt(base)));
    d = typecast2tc(pos_type, d);
    expr2tc c = add2tc(pos_type, d, constant_int2tc(pos_type, BigInt('0')));
    if (base == 16)
      c = if2tc(
        pos_type,
        lessthan2tc(d, constant_int2tc(pos_type, BigInt(10))),
        c,
        add2tc(
          pos_type,
          d,
          constant_int2tc(pos_type, BigInt((conv == 'X' ? 'A' : 'a') - 10))));
    digits.push_back(c);
  }
  return digits;
}

// What an integer conversion prints for `v`, already converted to the
// conversion's type, padded to `width` with spaces on the left, zeros after
// the sign (`zero`) or spaces on the right (`left`) (C11 7.21.6.1p6, p8).
static printf_charst
integer_chars(expr2tc v, char conv, bool left, bool zero, size_t width)
{
  if (v->type->get_width() < int_type2()->get_width())
    v = typecast2tc(int_type2(), v);
  const type2tc pos_type = int_type2();
  const type2tc utype = get_uint_type(v->type->get_width());
  const unsigned base = conv == 'o' ? 8 : conv == 'x' || conv == 'X' ? 16 : 10;
  const expr2tc neg =
    is_signedbv_type(v) ? lessthan2tc(v, gen_zero(v->type)) : gen_false_expr();
  const expr2tc as_unsigned = typecast2tc(utype, v);
  const expr2tc mag =
    if2tc(utype, neg, neg2tc(utype, as_unsigned), as_unsigned);
  size_t max_digits = 0;
  const expr2tc digits_len = digits_length(mag, base, max_digits);
  const std::vector<expr2tc> digits = digit_chars(mag, base, conv, max_digits);

  const expr2tc one = gen_one(pos_type), nought = gen_zero(pos_type);
  const expr2tc sign = if2tc(pos_type, neg, one, nought);
  const expr2tc core = add2tc(pos_type, sign, digits_len);
  const expr2tc w = constant_int2tc(pos_type, BigInt(width));
  const expr2tc pad =
    if2tc(pos_type, lessthan2tc(core, w), sub2tc(pos_type, w, core), nought);
  const bool zeros = zero && !left;
  const expr2tc sign_at = left || zeros ? nought : pad;
  const expr2tc first = left ? sign : add2tc(pos_type, pad, sign);
  const expr2tc filler = constant_int2tc(pos_type, BigInt(zeros ? '0' : ' '));
  return [=](const expr2tc &pos) {
    const expr2tc after_first = sub2tc(pos_type, pos, first);
    const expr2tc place_of =
      sub2tc(pos_type, sub2tc(pos_type, digits_len, one), after_first);
    const expr2tc in_digits = and2tc(
      greaterthanequal2tc(after_first, nought),
      lessthan2tc(after_first, digits_len));
    return if2tc(
      pos_type,
      and2tc(neg, equality2tc(pos, sign_at)),
      constant_int2tc(pos_type, BigInt('-')),
      if2tc(
        pos_type, in_digits, select_char(place_of, digits, filler), filler));
  };
}

// The most output characters stored as exact values rather than unknown
// nonzero ones; each costs a selection over every field.
static const size_t max_exact_output = 64;

// The `max_len` characters of `char_type` the fields print one after another,
// or none unless each field's characters are known.
static std::vector<expr2tc> output_chars(
  const std::vector<std::pair<expr2tc, printf_charst>> &fields,
  size_t max_len,
  const type2tc &char_type)
{
  std::vector<expr2tc> chars;
  if (
    max_len > max_exact_output ||
    !std::all_of(fields.begin(), fields.end(), [](const auto &field) {
      return bool(field.second);
    }))
    return chars;
  const type2tc pos_type = int_type2();
  for (size_t i = 0; i < max_len; i++)
  {
    const expr2tc pos = constant_int2tc(pos_type, BigInt(i));
    std::vector<std::pair<expr2tc, expr2tc>> cases;
    expr2tc start = gen_zero(pos_type);
    for (const auto &[field_len, field_chars] : fields)
    {
      const expr2tc rel = sub2tc(pos_type, pos, start);
      cases.emplace_back(
        and2tc(
          greaterthanequal2tc(rel, gen_zero(pos_type)),
          lessthan2tc(rel, field_len)),
        field_chars(rel));
      start = add2tc(pos_type, start, field_len);
    }
    expr2tc c = gen_zero(pos_type);
    for (size_t k = cases.size(); k-- > 0;)
      c = if2tc(pos_type, cases[k].first, cases[k].second, c);
    chars.push_back(typecast2tc(char_type, c));
  }
  return chars;
}

expr2tc goto_symext::string_length(
  const expr2tc &ptr,
  size_t &max_len,
  std::function<expr2tc(const expr2tc &)> *chars)
{
  expr2tc object;
  uint64_t offset, avail;
  if (
    !memcmp_resolve_operand(ptr, 0, object, offset, avail) || avail == 0 ||
    max_len + avail > max_stored_output || !is_array_type(object) ||
    !is_byte_type(to_array_type(object->type).subtype))
    return expr2tc();
  // The resolver drops NULL and invalid targets; this guard is false on them.
  expr2tc points_here = internal_deref_items.front().guard;
  cur_state->rename(points_here);
  do_simplify(points_here);

  // Off the object, or with no NUL in it (C11 7.24.1p1), the length is only
  // bounded the way the formatter bounds it.
  const type2tc len_type = int_type2();
  expr2tc unknown = gen_nondet(len_type);
  replace_nondet(unknown);
  assume(and2tc(
    greaterthanequal2tc(unknown, gen_zero(len_type)),
    lessthan2tc(unknown, constant_int2tc(len_type, BigInt(avail)))));

  const type2tc &elem = to_array_type(object->type).subtype;
  expr2tc len = unknown;
  for (uint64_t k = avail; k-- > 0;)
    len = if2tc(
      len_type,
      equality2tc(
        index2tc(elem, object, gen_ulong(offset + k)), gen_zero(elem)),
      constant_int2tc(len_type, BigInt(k)),
      len);
  max_len += avail - 1;
  if (chars && is_true(points_here))
    *chars = [object, offset, elem](const expr2tc &pos) {
      return typecast2tc(
        int_type2(),
        index2tc(
          elem,
          object,
          add2tc(
            size_type2(), gen_ulong(offset), typecast2tc(size_type2(), pos))));
    };
  return is_true(points_here) ? len
                              : if2tc(len_type, points_here, len, unknown);
}

expr2tc goto_symext::printf_field_length(
  const std::string &f,
  size_t &i,
  const std::vector<expr2tc> &args,
  size_t &arg,
  size_t &max_len,
  std::function<expr2tc(const expr2tc &)> &chars)
{
  const size_t spec = i;
  size_t width;
  bool left, zero;
  if (!parse_field_width(f, i, width, left, zero))
    return expr2tc();
  const size_t conv = i;
  while (i < f.size() && std::string("jlzt").find(f[i]) != std::string::npos)
    i++;
  if (i == f.size() || arg == args.size())
    return expr2tc();
  expr2tc v = args[arg++];
  expr2tc len;
  const size_t max_before = max_len;
  // A length modifier on %s makes it a wide string.
  if (f[i] == 's' && i == spec)
    len = string_length(v, max_len, &chars);
  else if (std::string("diuoxX").find(f[i]) != std::string::npos)
  {
    cur_state->rename(v);
    const std::string mod = f.substr(conv, i - conv);
    len = integer_length(v, f[i], mod, max_len);
    if (!is_nil_expr(len))
      chars = integer_chars(
        typecast2tc(printf_int_type(f[i] == 'd' || f[i] == 'i', mod), v),
        f[i],
        left,
        zero,
        width);
  }
  if (is_nil_expr(len))
    return expr2tc();
  return pad_to_width(len, width, max_before, max_len);
}

expr2tc goto_symext::printf_output_length(
  const code_printf2t &call,
  size_t fmt_idx,
  size_t &max_len,
  const type2tc &char_type,
  std::vector<expr2tc> *chars)
{
  expr2tc fmt = call.operands[fmt_idx];
  cur_state->rename(fmt);
  const expr2tc &fmt_str = get_base_object(fmt);
  if (!is_constant_string2t(fmt_str))
    return expr2tc();
  const std::string f = to_constant_string2t(fmt_str).value.as_string();

  // Each field's length, and the characters it prints when known.
  std::vector<std::pair<expr2tc, printf_charst>> fields;
  std::string text;
  size_t text_len = 0;
  const auto end_text = [&]() {
    if (!text.empty())
      fields.emplace_back(
        constant_int2tc(int_type2(), BigInt(text.size())), text_chars(text));
    text.clear();
  };
  max_len = 0;
  const std::vector<expr2tc> args = printf_args(call, fmt_idx);
  size_t arg = 0;
  for (size_t i = 0; i < f.size(); i++)
  {
    if (f[i] != '%' || (++i < f.size() && f[i] == '%'))
    {
      text += f[i];
      text_len++;
      continue;
    }
    end_text();
    printf_charst conv_chars;
    const expr2tc conv_len =
      printf_field_length(f, i, args, arg, max_len, conv_chars);
    if (is_nil_expr(conv_len))
      return expr2tc();
    fields.emplace_back(conv_len, conv_chars);
  }
  end_text();
  max_len += text_len;

  expr2tc len = gen_zero(int_type2());
  for (const auto &field : fields)
    len = add2tc(len->type, len, field.first);
  do_simplify(len);
  if (chars)
    for (expr2tc &c : *chars = output_chars(fields, max_len, char_type))
      do_simplify(c);
  return len;
}

expr2tc goto_symext::sprintf_output(
  const code_printf2t &call,
  size_t fmt_idx,
  const type2tc &char_type,
  std::vector<expr2tc> &chars)
{
  printf_formattert formatter;
  bool constant_args = false;
  if (!format_printf_output(call, fmt_idx, formatter, constant_args))
    return expr2tc();
  const std::string out = formatter.as_string();
  const bool sound = formatter.exact && formatter.bounded;
  if (constant_args && sound && formatter.min_outlen == formatter.max_outlen)
  {
    for (const char c : out)
      chars.push_back(constant_int2tc(
        char_type,
        is_signedbv_type(char_type) ? BigInt((signed char)c)
                                    : BigInt((unsigned char)c)));
    return constant_int2tc(int_type2(), BigInt(out.size()));
  }

  // Text, %s and integer conversions print no NUL, so a character not known
  // exactly is known to be nonzero.
  size_t max_len;
  expr2tc len = printf_output_length(call, fmt_idx, max_len, char_type, &chars);
  if (!is_nil_expr(len) && !chars.empty())
    return len;
  if (!is_nil_expr(len))
  {
    for (size_t i = 0; i < max_len; i++)
    {
      expr2tc c = gen_nondet(char_type);
      replace_nondet(c);
      assume(notequal2tc(c, gen_zero(char_type)));
      chars.push_back(c);
    }
    return len;
  }

  // Any other output the formatter bounds, up to max_stored_output: its length
  // is within those bounds and its characters may include a NUL (%c of 0).
  if (!sound || formatter.max_outlen > max_stored_output)
    return expr2tc();
  for (size_t i = 0; i < formatter.max_outlen; i++)
  {
    expr2tc c = gen_nondet(char_type);
    replace_nondet(c);
    chars.push_back(c);
  }
  len = gen_nondet(int_type2());
  replace_nondet(len);
  assume(and2tc(
    greaterthanequal2tc(
      len, gen_long(int_type2(), BigInt(formatter.min_outlen))),
    lessthanequal2tc(
      len, gen_long(int_type2(), BigInt(formatter.max_outlen)))));
  return len;
}

expr2tc goto_symext::symex_sprintf_store(const code_printf2t &call)
{
  const bool bounded_size =
    call.kind == printf_kindt::SNPRINTF || call.kind == printf_kindt::VSNPRINTF;
  if (
    !bounded_size && call.kind != printf_kindt::SPRINTF &&
    call.kind != printf_kindt::VSPRINTF)
    return expr2tc();

  const expr2tc &dst = call.operands[0];
  if (!is_pointer_type(dst) || !is_bv_type(to_pointer_type(dst->type).subtype))
    return expr2tc();
  const type2tc char_type = to_pointer_type(dst->type).subtype;
  // An output whose length is not soundly bounded leaves the destination
  // unchanged.
  std::vector<expr2tc> chars;
  const expr2tc len =
    sprintf_output(call, bounded_size ? 2 : 1, char_type, chars);
  if (is_nil_expr(len))
    return len;

  expr2tc n = bounded_size ? call.operands[1] : expr2tc();
  if (bounded_size)
    cur_state->rename(n);
  // Each write's guard joins the state guard, so the dereference checks are
  // scoped as well. C11 7.21.6.5p2: snprintf writes at most n - 1
  // characters, then a NUL.
  const guard2tc saved_guard = cur_state->guard;
  for (size_t i = 0; i <= chars.size(); i++)
  {
    const expr2tc pos = constant_int2tc(size_type2(), BigInt(i));
    const expr2tc at = constant_int2tc(len->type, BigInt(i));
    expr2tc written = lessthanequal2tc(at, len);
    expr2tc is_char =
      i < chars.size() ? lessthan2tc(at, len) : gen_false_expr();
    if (bounded_size)
    {
      written =
        and2tc(written, lessthan2tc(constant_int2tc(n->type, BigInt(i)), n));
      is_char = and2tc(
        is_char, lessthan2tc(constant_int2tc(n->type, BigInt(i + 1)), n));
    }
    do_simplify(written);
    do_simplify(is_char);
    guard2tc write_guard;
    write_guard.add(written);
    cur_state->guard = saved_guard;
    cur_state->guard.add(written);
    expr2tc value = if2tc(
      char_type,
      is_char,
      i < chars.size() ? chars[i] : gen_zero(char_type),
      gen_zero(char_type));
    do_simplify(value);
    const expr2tc elem = dereference2tc(char_type, add2tc(dst->type, dst, pos));
    symex_assign(code_assign2tc(elem, value), false, write_guard);
  }
  cur_state->guard = saved_guard;
  return len;
}

expr2tc goto_symext::printf_return_length(
  const expr2tc &lhs,
  const code_printf2t &call,
  size_t fmt_idx)
{
  const expr2tc stored = symex_sprintf_store(call);
  if (!is_nil_expr(stored) || is_nil_expr(lhs))
    return stored;
  // asprintf returns -1 when allocation fails, and a v* call's arguments are
  // behind its va_list, so only the direct stream forms return the length.
  switch (call.kind)
  {
  case printf_kindt::PRINTF:
  case printf_kindt::FPRINTF:
  case printf_kindt::DPRINTF:
    break;
  case printf_kindt::SPRINTF:
  case printf_kindt::SNPRINTF:
  case printf_kindt::ASPRINTF:
  case printf_kindt::VPRINTF:
  case printf_kindt::VFPRINTF:
  case printf_kindt::VSPRINTF:
  case printf_kindt::VSNPRINTF:
  case printf_kindt::VASPRINTF:
    return expr2tc();
  }
  size_t max_len;
  return printf_output_length(call, fmt_idx, max_len);
}

void goto_symext::assume_printf_length(
  const expr2tc &retval,
  const expr2tc &len)
{
  if (!is_nil_expr(retval) && !is_nil_expr(len))
    assume(equality2tc(retval, len));
}

void goto_symext::assume_printf_range(
  const expr2tc &retval,
  const BigInt &lo,
  const BigInt &hi,
  bool may_fail)
{
  expr2tc in_range = and2tc(
    greaterthanequal2tc(retval, constant_int2tc(retval->type, lo)),
    lessthanequal2tc(retval, constant_int2tc(retval->type, hi)));
  // asprintf still returns -1 when its allocation fails.
  if (may_fail)
    in_range =
      or2tc(in_range, equality2tc(retval, constant_int2tc(retval->type, -1)));
  assume(in_range);
}

void goto_symext::symex_asprintf_buffer(
  const expr2tc &strp,
  const expr2tc &retval)
{
  // Only the success path allocates. POSIX leaves *strp undefined when the
  // call fails, so the caller owns nothing and correctly returns without
  // freeing; allocating unconditionally orphans the buffer on that path and
  // trips --memory-leak-check (GitHub #6966). The guard reaches the object's
  // allocation guard via symex_mem, so the leak check is scoped too.
  guard2tc alloc_guard;
  expr2tc alloc_size = constant_int2tc(size_type2(), BigInt(1));
  if (!is_nil_expr(retval))
  {
    alloc_guard.add(greaterthanequal2tc(retval, gen_zero(retval->type)));
    alloc_size =
      add2tc(size_type2(), typecast2tc(size_type2(), retval), alloc_size);
  }

  // Derive char * from strp's declared type (char **) so the dereference
  // width matches what the value-set analysis and SMT encoding expect.
  type2tc char_ptr_type = to_pointer_type(strp->type).subtype;
  expr2tc deref_strp = dereference2tc(char_ptr_type, strp);
  expr2tc malloc_se = sideeffect2tc(
    char_ptr_type,
    expr2tc(),
    alloc_size,
    std::vector<expr2tc>(),
    char_type2(),
    sideeffect2t::allockind::malloc);
  symex_assign(code_assign2tc(deref_strp, malloc_se), false, alloc_guard);

  if (is_nil_expr(retval))
    return;

  // A non-negative return means the allocation succeeded, so the buffer
  // cannot be the NULL that malloc may hand back.
  expr2tc buf = dereference2tc(char_ptr_type, strp);
  dereference(buf, dereferencet::READ);
  assume(or2tc(
    lessthan2tc(retval, gen_zero(retval->type)),
    notequal2tc(buf, symbol2tc(char_ptr_type, "NULL"))));
}

// The formatter computes the return value, and also the %n counts, which are
// written even when the result is discarded.
static bool printf_needs_formatter(
  const expr2tc &lhs,
  bool format_is_constant,
  const irep_idt &fmt)
{
  return !is_nil_expr(lhs) ||
         (format_is_constant && fmt.as_string().find('n') != std::string::npos);
}

void goto_symext::fold_printf_char_arrays(std::list<expr2tc> &args)
{
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
}

void goto_symext::assign_printf_percent_n(const printf_formattert &formatter)
{
  // Behind an unrecovered va_list the operand is not the %n pointer.
  if (!formatter.args_reliable)
    return;
  for (const printf_formattert::percent_nt &pn : formatter.percent_n)
  {
    if (!is_pointer_type(pn.target->type))
      continue;
    const type2tc &count_type = to_pointer_type(pn.target->type).subtype;
    if (!is_bv_type(count_type))
      continue;
    expr2tc count;
    if (pn.bounded && pn.min_outlen == pn.max_outlen)
      count = constant_int2tc(count_type, BigInt(pn.min_outlen));
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
      if (pn.bounded)
        assume_printf_range(nondet, pn.min_outlen, pn.max_outlen, false);
      else
        assume(greaterthanequal2tc(nondet, gen_zero(int_type2())));
      count = typecast2tc(count_type, nondet);
    }
    symex_assign(code_assign2tc(dereference2tc(count_type, pn.target), count));
  }
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

  // Check format specifiers against original operands before
  // renaming/conversion
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

  const expr2tc output_len =
    printf_return_length(lhs, to_code_printf2t(rhs), fmt_idx);

  // Now we pop the format
  for (size_t i = 0; i < idx; i++)
    new_rhs.operands.erase(new_rhs.operands.begin());

  std::list<expr2tc> args;
  new_rhs.foreach_operand([this, &args](const expr2tc &e) {
    expr2tc tmp = e;
    do_simplify(tmp);
    args.push_back(tmp);
  });

  // The value assigned to the return lvalue, nil when the result is discarded.
  // Needed below to keep the asprintf/vasprintf *strp allocation off the
  // failure path.
  expr2tc retval;

  if (printf_needs_formatter(lhs, format_is_constant, fmt))
  {
    // get the return value from code_printf2tc
    // 1. covert code_printf2tc back to sideeffect2tc
    if (!is_nil_expr(lhs))
    {
      exprt rhs_expr = migrate_expr_back(rhs);
      exprt printf_code("sideeffect", migrate_type_back(lhs->type));

      printf_code.statement("printf2");
      printf_code.operands() = rhs_expr.operands();
      printf_code.location() = rhs_expr.location();

      migrate_expr(printf_code, rhs);
    }

    // With a successful va_list recovery the single va_list operand is
    // replaced by the actual variadic arguments, restoring the direct-call
    // argument shape the formatter expects.
    if (va_recovered)
      args = recovered_args;

    // 2 printf_formatter does not handle char arrays: fold them to strings.
    fold_printf_char_arrays(args);

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
    assign_printf_percent_n(printf_formatter);

    if (!is_nil_expr(lhs))
    {
      const bool sound_bound = format_is_constant && printf_formatter.bounded;
      if (
        sound_bound && !is_allocating &&
        printf_formatter.min_outlen == printf_formatter.max_outlen)
      {
        retval =
          constant_int2tc(int_type2(), BigInt(printf_formatter.max_outlen));
        symex_assign(code_assign2tc(lhs, retval));
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
        retval = nondet;
        if (sound_bound)
          assume_printf_range(
            nondet,
            printf_formatter.min_outlen,
            printf_formatter.max_outlen,
            is_allocating);
        else if (is_allocating)
          // -1 on allocation failure; cap success side at INT_MAX/2.
          assume_printf_range(nondet, -1, INT_MAX / 2, false);
        else
          assume(greaterthanequal2tc(
            nondet, constant_int2tc(int_type2(), BigInt(0))));
        symex_assign(code_assign2tc(lhs, nondet));
      }
    }
  }

  assume_printf_length(retval, output_len);

  if (is_allocating && !is_nil_expr(strp) && is_pointer_type(strp->type))
    symex_asprintf_buffer(strp, retval);

  target->output(
    cur_state->guard.as_expr(), cur_state->source, fmt.as_string(), args);
}

// One entry per conversion that takes an argument: true if it assigns an
// input item, false for %n, which the return value does not count.
static std::vector<bool> scanf_conversions(const std::string &fmt)
{
  std::vector<bool> conversions;
  for (size_t i = 0; i + 1 < fmt.length(); ++i)
  {
    if (fmt[i] != '%')
      continue;
    if (fmt[++i] == '%')
      continue;

    // Width and length modifiers. '*' is left out: a suppressed conversion
    // takes no argument.
    while (i < fmt.length() &&
           std::string("-+ #0123456789.hlLzjt").find(fmt[i]) !=
             std::string::npos)
      ++i;

    if (
      i < fmt.length() &&
      std::string("diouxXfFeEgGaAcsp[n").find(fmt[i]) != std::string::npos)
      conversions.push_back(fmt[i] != 'n');
  }
  return conversions;
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

  expr2tc fmt_operand = func_call.operands[fmt_idx];
  cur_state->rename(fmt_operand);

  const size_t available_args = func_call.operands.size() - (fmt_idx + 1);
  std::vector<bool> conversions(available_args, true);
  const expr2tc &base_expr = get_base_object(fmt_operand);
  if (is_constant_string2t(base_expr))
    conversions =
      scanf_conversions(to_constant_string2t(base_expr).value.as_string());
  conversions.resize(std::min(conversions.size(), available_args));

  // C11 7.21.6.2p16: EOF if input fails before the first conversion, else the
  // number of items assigned, which a matching failure can leave below the
  // number of conversions.
  const long items = std::count(conversions.begin(), conversions.end(), true);
  expr2tc count = gen_nondet(int_type2());
  replace_nondet(count);
  assume(and2tc(
    greaterthanequal2tc(count, gen_long(int_type2(), -1)),
    lessthanequal2tc(count, gen_long(int_type2(), items))));
  if (func_call.ret)
    symex_assign(code_assign2tc(func_call.ret, count));

  // TODO: fill / cut off the inputs stream based on the length limits.

  long assigned = 0;
  for (size_t i = 0; i < conversions.size(); i++)
  {
    // Item k is stored only when the k conversions before it matched too,
    // and a %n target only when every item before it was assigned.
    guard2tc stored;
    if (conversions[i])
      stored.add(greaterthan2tc(count, gen_long(int_type2(), assigned++)));
    else if (assigned > 0)
      stored.add(greaterthanequal2tc(count, gen_long(int_type2(), assigned)));

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

      symex_assign(code_assign2tc(item.object, val), false, stored);
    }
  }

  cur_state->source.pc++;
}
