#include <goto-symex/trace/printf_formatter.h>
#include <sstream>
#include <util/lang/c_types.h>
#include <util/config/config.h>
#include <irep2/irep2_utils.h>
#include <util/arith/format_constant.h>
#include <util/arith/mp_arith.h>
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

std::string pad_field(const std::string &s, unsigned min_width, bool left)
{
  if (s.length() >= min_width)
    return s;
  const std::string fill(min_width - s.length(), ' ');
  return left ? s + fill : fill + s;
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

// Maximum number of decimal digits of a bits-wide magnitude.
size_t max_decimal_digits(size_t bits)
{
  // ceil(bits * log10(2)) via integer arithmetic (30103/100000 ≈ log10(2))
  return (bits * 30103 + 99999) / 100000;
}

// The flags of a conversion specification that format_spect does not hold.
struct field_flagst
{
  bool left = false;
  bool plus = false;
  bool space = false;
  bool alternate = false;
  bool has_precision = false;

  // format_constant renders no flag but '0' for a floating-point conversion.
  bool float_length_modelled() const
  {
    return !plus && !space && !alternate;
  }

  // Record ch if it is a flag character other than '0'.
  bool parse(char ch)
  {
    switch (ch)
    {
    case '-':
      left = true;
      return true;
    case '+':
      plus = true;
      return true;
    case ' ':
      space = true;
      return true;
    case '#':
      alternate = true;
      return true;
    default:
      return false;
    }
  }
};

// The digits of an integer conversion, padded to the precision.
std::string int_digits(
  const BigInt &value,
  int base,
  bool uppercase,
  const field_flagst &flags,
  unsigned precision)
{
  std::string digits = (base == 10)
                         ? integer2string(value.is_negative() ? -value : value)
                         : format_radix(value.to_uint64(), base, uppercase);
  if (flags.has_precision)
  {
    if (value.is_zero() && precision == 0)
      digits.clear();
    if (digits.length() < precision)
      digits.insert(0, precision - digits.length(), '0');
  }
  if (flags.alternate && base == 8 && (digits.empty() || digits[0] != '0'))
    digits.insert(0, "0");
  return digits;
}

// The sign or radix prefix of an integer conversion.
std::string int_prefix(
  const BigInt &value,
  int base,
  bool uppercase,
  bool is_signed,
  const field_flagst &flags)
{
  if (value.is_negative())
    return "-";
  if (flags.alternate && base == 16 && !value.is_zero())
    return uppercase ? "0X" : "0x";
  if (!is_signed)
    return "";
  return flags.plus ? "+" : flags.space ? " " : "";
}

// The text of an integer conversion (C11 7.21.6.1p6, p8).
std::string layout_int(
  const BigInt &value,
  int base,
  bool uppercase,
  bool is_signed,
  const field_flagst &flags,
  const format_spect &spec)
{
  std::string digits =
    int_digits(value, base, uppercase, flags, spec.precision);
  const std::string prefix =
    int_prefix(value, base, uppercase, is_signed, flags);
  // '0' is ignored under '-' or an explicit precision.
  const size_t length = prefix.length() + digits.length();
  if (
    spec.zero_padding && !flags.left && !flags.has_precision &&
    length < spec.min_width)
    digits.insert(0, spec.min_width - length, '0');
  return pad_field(prefix + digits, spec.min_width, flags.left);
}

// The longest text an integer conversion of a bits-wide argument can produce.
size_t max_int_chars(
  size_t bits,
  int base,
  bool is_signed,
  const field_flagst &flags,
  const format_spect &spec)
{
  const size_t digits = (base == 16) ? (bits + 3) / 4
                        : (base == 8)
                          ? (bits + 2) / 3
                          : max_decimal_digits(is_signed ? bits - 1 : bits);
  const size_t prefix = is_signed          ? 1
                        : !flags.alternate ? 0
                        : (base == 16)     ? 2
                        : (base == 8)      ? 1
                                           : 0;
  const size_t precision = flags.has_precision ? spec.precision : 0;
  return std::max<size_t>(prefix + std::max(digits, precision), spec.min_width);
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
  format_constantt format_constant;

  format_constant.precision = 6;
  format_constant.min_width = 0;
  format_constant.zero_padding = false;

  char ch = next();

  field_flagst flags;
  for (; flags.parse(ch) || ch == '0'; ch = next())
    format_constant.zero_padding |= ch == '0';

  while (isdigit(ch)) // width
  {
    format_constant.min_width *= 10;
    format_constant.min_width += ch - '0';
    ch = next();
  }

  if (ch == '.') // precision
  {
    flags.has_precision = true;
    format_constant.precision = 0;
    ch = next();

    while (isdigit(ch))
    {
      format_constant.precision *= 10;
      format_constant.precision += ch - '0';
      ch = next();
    }
  }

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
  // record [min, max] over every value of the argument's type.
  auto emit_int = [&](bool is_signed, int base, bool uppercase) {
    if (next_operand == operands.end())
    {
      // Expected an integer argument but none is available: cannot bound.
      bounded = false;
      return;
    }
    const type2tc target = pick_int_type(is_signed, length_mod);
    const expr2tc casted = make_type(*(next_operand++), target);
    std::string s;
    size_t min_chars, max_chars;
    if (is_constant_int2t(casted))
    {
      s = layout_int(
        to_constant_int2t(casted).value,
        base,
        uppercase,
        is_signed,
        flags,
        format_constant);
      min_chars = max_chars = s.length();
    }
    else
    {
      max_chars = max_int_chars(
        target->get_width(), base, is_signed, flags, format_constant);
      s = std::string(max_chars, '0');
      min_chars = std::max<size_t>(
        format_constant.min_width,
        flags.has_precision ? format_constant.precision : 1);
    }
    out << s;
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
    bounded &= flags.float_length_modelled();
    // format_constant pads with spaces on the left only.
    exact &= !flags.left && !format_constant.zero_padding;
    if (!bounded)
      return;
    if (
      is_constant_floatbv2t(farg) || is_constant_fixedbv2t(farg) ||
      is_constant_int2t(farg))
      emit(format_constant(farg));
    else
      bounded = false;
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
    const expr2tc &op = *(next_operand++);
    const expr2tc symbol2 = get_base_object(op);
    exprt char_array = migrate_expr_back(symbol2);
    if (char_array.id() == "string-constant")
    {
      const std::string &str = char_array.value().as_string();
      emit(pad_field(
        flags.has_precision ? str.substr(0, format_constant.precision) : str,
        format_constant.min_width,
        flags.left));
      break;
    }
    // A non-literal %s: derive a sound upper bound from the pointed-to
    // object's size when possible. If the argument points into a constant-size
    // char array of N bytes, a valid C string there has strlen <= N-1 (the NUL
    // must fit), and that holds for any starting offset within the array.
    // Contribute N-1 to the maximum (the string may be empty, so the minimum is
    // unchanged) and keep the result bounded. The restriction to a finite,
    // 8-bit-element array keeps the byte size well-defined without a namespace
    // and matches the C-string narrative; anything else (a bare pointer of
    // unknown extent, an incomplete/VLA array, a non-char element type) has no
    // statically-known bound, so the output is treated as unbounded.
    if (
      args_reliable && is_array_type(symbol2->type) &&
      !array_or_vector_size_is_infinite(symbol2->type) &&
      is_byte_type(to_array_type(symbol2->type).subtype))
    {
      const BigInt nbytes = type_byte_size_default(symbol2->type, BigInt(0));
      if (nbytes > 0)
      {
        size_t max_len = (nbytes - 1).to_uint64();
        if (flags.has_precision)
          max_len = std::min(max_len, size_t(format_constant.precision));
        min_outlen += format_constant.min_width;
        max_outlen += std::max(max_len, size_t(format_constant.min_width));
        break;
      }
    }
    bounded = false;
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
    if (is_constant_int2t(carg))
      emit(pad_field(
        std::string(1, (char)to_constant_int2t(carg).value.to_int64()),
        format_constant.min_width,
        flags.left));
    else
    {
      // %c writes exactly one character regardless of its value (padded to
      // min_width). The length is value-independent, so this stays bounded.
      const size_t clen =
        format_constant.min_width > 1 ? format_constant.min_width : 1;
      emit(std::string(clen, ' '));
    }
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
    emit(pad_field(
      "0x" + std::string(hex_chars, '0'),
      format_constant.min_width,
      flags.left));
    break;
  }

  default:
    exact = false;
    emit(std::string(1, '%') + ch);
  }
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
