#include <goto-symex/trace/printf_formatter.h>
#include <optional>
#include <sstream>
#include <util/lang/c_types.h>
#include <util/config/config.h>
#include <irep2/irep2_utils.h>
#include <util/arith/format_constant.h>
#include <util/expr/type_byte_size.h>

const expr2tc
printf_formattert::make_type(const expr2tc &src, const type2tc &dest)
{
  // Callers test the result with is_constant_int2t, so fold on both paths --
  // an operand already of the target type may be constant only after folding.
  expr2tc tmp = src;
  if (src->type != dest)
    tmp = typecast2tc(dest, src);
  simplify(tmp);
  return tmp;
}

void printf_formattert::operator()(
  const std::string &_format,
  const std::list<expr2tc> &_operands)
{
  format = _format;
  operands = _operands;
}

void printf_formattert::print(std::ostream &out)
{
  format_pos = 0;
  next_operand = operands.begin();
  min_outlen = 0;
  max_outlen = 0;
  bounded = true;
  exact = true;

  try
  {
    while (!eol())
      process_char(out);
  }

  catch (eol_exception)
  {
  }
}

std::string printf_formattert::as_string()
{
  std::ostringstream stream;
  print(stream);
  return stream.str();
}

namespace
{
enum class length_modt
{
  NONE,
  HH,
  H,
  L,
  LL,
  J,
  Z,
  T,
  BIG_L,
};

struct conversiont
{
  bool left = false;
  bool plus = false;
  bool space = false;
  bool alt = false;
  bool zero = false;
  unsigned width = 0;
  std::optional<unsigned> precision;
};

// Pad to the field width (C11 7.21.6.1p4-6); zero padding goes after the
// sign or radix prefix.
std::string pad_field(
  const std::string &prefix,
  const std::string &body,
  const conversiont &conv,
  bool zero)
{
  const size_t len = prefix.length() + body.length();
  if (len >= conv.width)
    return prefix + body;
  const std::string fill(conv.width - len, zero && !conv.left ? '0' : ' ');
  if (conv.left)
    return prefix + body + fill;
  if (zero)
    return prefix + fill + body;
  return fill + prefix + body;
}

std::string sign_prefix(bool negative, bool is_signed, const conversiont &conv)
{
  if (negative)
    return "-";
  if (is_signed && conv.plus)
    return "+";
  if (is_signed && conv.space)
    return " ";
  return "";
}

// Render an unsigned 64-bit value in base 8 or 16.
std::string format_radix(uint64_t val, int base, bool uppercase)
{
  std::ostringstream oss;
  if (base == 16 && uppercase)
    oss << std::uppercase;
  oss << (base == 16 ? std::hex : std::oct) << val;
  return oss.str();
}

// The digits of |value|, at least as many as the precision (C11
// 7.21.6.1p8).
std::string int_digits(
  const BigInt &value,
  int base,
  bool uppercase,
  const conversiont &conv)
{
  if (conv.precision == 0u && value.is_zero())
    return "";
  std::string digits = base == 10
                         ? integer2string(value.is_negative() ? -value : value)
                         : format_radix(value.to_uint64(), base, uppercase);
  if (conv.precision && digits.length() < *conv.precision)
    digits.insert(0, *conv.precision - digits.length(), '0');
  return digits;
}

std::string format_int(
  const BigInt &value,
  int base,
  bool uppercase,
  bool is_signed,
  const conversiont &conv)
{
  const bool negative = value.is_negative() && !value.is_zero();
  std::string digits = int_digits(value, base, uppercase, conv);
  std::string prefix = sign_prefix(negative, is_signed, conv);
  if (conv.alt && base == 8 && (digits.empty() || digits[0] != '0'))
    digits.insert(0, "0");
  if (conv.alt && base == 16 && !value.is_zero())
    prefix += uppercase ? "0X" : "0x";
  return pad_field(prefix, digits, conv, conv.zero && !conv.precision);
}

// Maximum number of digits of a bits-wide integer in base 8, 10 or 16.
size_t max_digits(size_t bits, int base, bool is_signed)
{
  if (base == 16)
    return (bits + 3) / 4;
  if (base == 8)
    return (bits + 2) / 3;
  const size_t mag_bits = is_signed ? bits - 1 : bits;
  // ceil(mag_bits * log10(2)) via integer arithmetic (30103/100000 ≈ log10(2))
  return (mag_bits * 30103 + 99999) / 100000;
}

// Bounds on the length of an integer conversion of an unknown bits-wide value.
std::pair<size_t, size_t> int_length_bounds(
  size_t bits,
  int base,
  bool is_signed,
  const conversiont &conv)
{
  const size_t mag_digits = max_digits(bits, base, is_signed);
  const bool octal_alt = conv.alt && base == 8;
  const size_t precision = conv.precision.value_or(1);
  size_t min_len = octal_alt ? std::max(precision, size_t(1)) : precision;
  size_t max_len = std::max(mag_digits, precision) + octal_alt;
  if (is_signed)
  {
    min_len += conv.plus || conv.space;
    max_len++;
  }
  if (conv.alt && base == 16)
    max_len += 2;
  return {
    std::max(min_len, size_t(conv.width)),
    std::max(max_len, size_t(conv.width))};
}

// A non-literal %s: derive a sound upper bound from the pointed-to object's
// size when possible. If the argument points into a constant-size char array
// of N bytes, a valid C string there has strlen <= N-1 (the NUL must fit), and
// that holds for any starting offset within the array. The restriction to a
// finite, 8-bit-element array keeps the byte size well-defined without a
// namespace; anything else (a bare pointer of unknown extent, an
// incomplete/VLA array, a non-char element type) has no statically-known
// bound.
std::optional<size_t> string_length_bound(const expr2tc &object)
{
  if (
    !is_array_type(object->type) ||
    array_or_vector_size_is_infinite(object->type) ||
    !is_byte_type(to_array_type(object->type).subtype))
    return std::nullopt;
  const BigInt nbytes = type_byte_size_default(object->type, BigInt(0));
  if (nbytes <= 0)
    return std::nullopt;
  return (nbytes - 1).to_uint64();
}

// Pick the integer cast target for a given signedness and length modifier.
type2tc pick_int_type(bool is_signed, length_modt mod)
{
  if (is_signed)
  {
    switch (mod)
    {
    case length_modt::HH:
      return get_int8_type();
    case length_modt::H:
      return get_int16_type();
    case length_modt::L:
      return long_int_type2();
    case length_modt::Z:
    case length_modt::T:
      return signed_size_type2();
    case length_modt::J:
    case length_modt::LL:
      return long_long_int_type2();
    default:
      return int_type2();
    }
  }
  switch (mod)
  {
  case length_modt::HH:
    return get_uint8_type();
  case length_modt::H:
    return get_uint16_type();
  case length_modt::L:
    return long_uint_type2();
  case length_modt::Z:
    return size_type2();
  case length_modt::T:
    return signed_size_type2();
  case length_modt::J:
  case length_modt::LL:
    return long_long_uint_type2();
  default:
    return uint_type2();
  }
}
} // namespace

void printf_formattert::process_format(std::ostream &out)
{
  conversiont conv;
  char ch = next();

  while (ch == '0' || ch == '-' || ch == '+' || ch == ' ' || ch == '#')
  {
    conv.zero |= ch == '0';
    conv.left |= ch == '-';
    conv.plus |= ch == '+';
    conv.space |= ch == ' ';
    conv.alt |= ch == '#';
    ch = next();
  }

  while (isdigit(ch)) // width
  {
    conv.width = conv.width * 10 + (ch - '0');
    ch = next();
  }

  if (ch == '.') // precision
  {
    conv.precision = 0;
    ch = next();

    while (isdigit(ch))
    {
      *conv.precision = *conv.precision * 10 + (ch - '0');
      ch = next();
    }
  }

  format_constantt format_constant;
  format_constant.precision = conv.precision.value_or(6);

  // Parse length modifier: hh, h, ll, l, L, z, j, t
  length_modt length_mod = length_modt::NONE;
  if (ch == 'h')
  {
    ch = next();
    if (ch == 'h')
    {
      length_mod = length_modt::HH;
      ch = next();
    }
    else
      length_mod = length_modt::H;
  }
  else if (ch == 'l')
  {
    ch = next();
    if (ch == 'l')
    {
      length_mod = length_modt::LL;
      ch = next();
    }
    else
      length_mod = length_modt::L;
  }
  else if (ch == 'L')
  {
    length_mod = length_modt::BIG_L;
    ch = next();
  }
  else if (ch == 'z')
  {
    length_mod = length_modt::Z;
    ch = next();
  }
  else if (ch == 'j')
  {
    length_mod = length_modt::J;
    ch = next();
  }
  else if (ch == 't')
  {
    length_mod = length_modt::T;
    ch = next();
  }

  // Emit a string of known length and update both output-length bounds.
  auto emit = [&](const std::string &s) {
    out << s;
    min_outlen += s.length();
    max_outlen += s.length();
  };

  // Emit an integer in the given base and update output-length bounds.
  // For constant args the exact formatted length is known. For non-constant
  // args we emit a max-width zero placeholder for the counterexample and
  // record the bounds from int_length_bounds.
  auto emit_int = [&](bool is_signed, int base, bool uppercase) {
    if (next_operand == operands.end())
    {
      // Expected an integer argument but none is available: cannot bound.
      bounded = false;
      return;
    }
    const type2tc target = pick_int_type(is_signed, length_mod);
    const expr2tc casted = make_type(*(next_operand++), target);
    if (is_constant_int2t(casted))
    {
      emit(format_int(
        to_constant_int2t(casted).value, base, uppercase, is_signed, conv));
      return;
    }
    const auto [min_chars, max_chars] =
      int_length_bounds(target->get_width(), base, is_signed, conv);
    out << std::string(max_chars, '0');
    min_outlen += min_chars;
    max_outlen += max_chars;
  };

  // Emit a floating-point conversion (%e/%f/%g). Only a constant argument has
  // a statically-known rendering; a non-constant double can format to an
  // arbitrarily long string (e.g. %f of 1e308), so we cannot bound the output
  // and mark the whole result unbounded. The operand is still consumed to keep
  // subsequent arguments aligned.
  auto emit_float = [&](format_spect::stylet style) {
    format_constant.style = style;
    if (next_operand == operands.end())
    {
      bounded = false;
      return;
    }
    const expr2tc farg = make_type(*(next_operand++), double_type2());
    if (
      !is_constant_floatbv2t(farg) && !is_constant_fixedbv2t(farg) &&
      !is_constant_int2t(farg))
    {
      bounded = false;
      return;
    }
    const std::string s = format_constant(farg);
    const bool negative = s[0] == '-';
    const std::string body = negative ? s.substr(1) : s;
    emit(pad_field(
      sign_prefix(negative, true, conv),
      body,
      conv,
      conv.zero && isdigit(body[0])));
  };

  switch (ch)
  {
  case '%':
    emit(std::string(1, ch));
    break;

  case 'e':
  case 'E':
    emit_float(format_spect::stylet::SCIENTIFIC);
    break;

  case 'f':
  case 'F':
    emit_float(format_spect::stylet::DECIMAL);
    break;

  case 'g':
  case 'G':
    if (format_constant.precision == 0)
      format_constant.precision = 1;
    emit_float(format_spect::stylet::AUTOMATIC);
    break;

  case 's':
  {
    if (next_operand == operands.end())
    {
      // A %s with no argument to inspect: length is unknown, so the output
      // has no sound upper bound.
      bounded = false;
      break;
    }
    const expr2tc symbol2 = get_base_object(*(next_operand++));
    exprt char_array = migrate_expr_back(symbol2);
    if (char_array.id() == "string-constant")
    {
      const std::string str = char_array.value().as_string();
      emit(pad_field(
        "", str.substr(0, conv.precision.value_or(str.size())), conv, false));
      break;
    }
    std::optional<size_t> max_len =
      args_reliable ? string_length_bound(symbol2) : std::nullopt;
    if (conv.precision)
      max_len =
        std::min(max_len.value_or(*conv.precision), size_t(*conv.precision));
    if (!max_len)
    {
      bounded = false;
      break;
    }
    min_outlen += conv.width;
    max_outlen += std::max(*max_len, size_t(conv.width));
  }
  break;

  case 'i':
  case 'd':
    emit_int(true, 10, false);
    break;

  case 'D':
    // Legacy BSD: %D is %ld
    if (length_mod == length_modt::NONE)
      length_mod = length_modt::L;
    emit_int(true, 10, false);
    break;

  case 'u':
    emit_int(false, 10, false);
    break;

  case 'U':
    // Legacy BSD: %U is %lu
    if (length_mod == length_modt::NONE)
      length_mod = length_modt::L;
    emit_int(false, 10, false);
    break;

  case 'c':
  {
    if (next_operand == operands.end())
    {
      bounded = false;
      break;
    }
    const expr2tc carg = make_type(*(next_operand++), char_type2());
    // %c writes exactly one character regardless of its value, so a
    // non-constant one stays bounded.
    const char c =
      is_constant_int2t(carg)
        ? static_cast<char>(to_constant_int2t(carg).value.to_int64())
        : ' ';
    emit(pad_field("", std::string(1, c), conv, false));
    break;
  }

  case 'x':
    emit_int(false, 16, false);
    break;

  case 'X':
    emit_int(false, 16, true);
    break;

  case 'o':
    emit_int(false, 8, false);
    break;

  case 'p':
  {
    // Consume the pointer argument and emit a placeholder sized to the
    // target's pointer width (e.g. "0x" + 16 hex digits on 64-bit). Using a
    // plausible length keeps printf's modelled return value close to the
    // runtime output. %p ignores the '0' flag; pad with spaces only.
    if (next_operand != operands.end())
      ++next_operand;
    exact = false;
    const unsigned hex_chars = (config.ansi_c.pointer_width() + 3) / 4;
    emit(pad_field("", "0x" + std::string(hex_chars, '0'), conv, false));
    break;
  }

  default:
    exact = false;
    emit(std::string(1, '%') + ch);
  }

  // '#' is defined only for o, x, X and the floating conversions, and is
  // modelled only for the first three.
  if (conv.alt && ch != 'o' && ch != 'x' && ch != 'X')
    exact = false;
}

void printf_formattert::process_char(std::ostream &out)
{
  char ch = next();

  if (ch == '%')
    process_format(out);
  else
  {
    out << ch;
    min_outlen++;
    max_outlen++;
  }
}
