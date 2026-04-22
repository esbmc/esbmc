#include <goto-symex/printf_formatter.h>
#include <sstream>
#include <util/c_types.h>
#include <util/config.h>
#include <irep2/irep2_utils.h>
#include <util/format_constant.h>
#include <util/type_byte_size.h>

const expr2tc
printf_formattert::make_type(const expr2tc &src, const type2tc &dest)
{
  if (src->type == dest)
    return src;

  expr2tc tmp = typecast2tc(dest, src);
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

std::string
pad_int(const std::string &s, unsigned min_width, bool zero_padding)
{
  if (s.length() >= min_width)
    return s;
  if (zero_padding && s[0] == '-')
    return s[0] + std::string(min_width - s.length(), '0') + s.substr(1);
  return std::string(min_width - s.length(), zero_padding ? '0' : ' ') + s;
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
    case length_modt::Z:
    case length_modt::J:
    case length_modt::T:
      return long_int_type2();
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
  case length_modt::Z:
  case length_modt::J:
  case length_modt::T:
    return long_uint_type2();
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

  // Parse flags: only '0' (zero-pad) is tracked; others are consumed
  while (ch == '0' || ch == '-' || ch == '+' || ch == ' ' || ch == '#')
  {
    if (ch == '0')
      format_constant.zero_padding = true;
    ch = next();
  }

  while (isdigit(ch)) // width
  {
    format_constant.min_width *= 10;
    format_constant.min_width += ch - '0';
    ch = next();
  }

  if (ch == '.') // precision
  {
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

  // Emit an integer in the given base. Constants are rendered via
  // format_constantt (decimal) or format_radix (hex/oct). For non-constant
  // hex/oct args we fall back to a zero-string sized to the maximum number
  // of digits the target type can produce — this keeps counterexample
  // output readable while providing a conservative upper bound on the
  // output length so printf's modelled return value never silently
  // under-counts. Decimal non-constants use format_constantt's existing
  // failure-string fallback to preserve pre-existing behaviour.
  auto emit_int = [&](bool is_signed, int base, bool uppercase) {
    if (next_operand == operands.end())
      return;
    const type2tc target = pick_int_type(is_signed, length_mod);
    const expr2tc casted = make_type(*(next_operand++), target);
    std::string s;
    if (is_constant_int2t(casted))
    {
      if (base == 10)
        s = format_constant(casted);
      else
        s = format_radix(
          to_constant_int2t(casted).value.to_uint64(), base, uppercase);
    }
    else if (base == 10)
      s = format_constant(casted);
    else
    {
      const size_t bits = target->get_width();
      const size_t max_digits =
        base == 16 ? (bits + 3) / 4 : (bits + 2) / 3;
      s = std::string(max_digits, '0');
    }
    out << pad_int(s, format_constant.min_width, format_constant.zero_padding);
  };

  switch (ch)
  {
  case '%':
    out << ch;
    break;

  case 'e':
  case 'E':
    format_constant.style = format_spect::stylet::SCIENTIFIC;
    if (next_operand == operands.end())
      break;
    out << format_constant(make_type(*(next_operand++), double_type2()));
    break;

  case 'f':
  case 'F':
    format_constant.style = format_spect::stylet::DECIMAL;
    if (next_operand == operands.end())
      break;
    out << format_constant(make_type(*(next_operand++), double_type2()));
    break;

  case 'g':
  case 'G':
    format_constant.style = format_spect::stylet::AUTOMATIC;
    if (format_constant.precision == 0)
      format_constant.precision = 1;
    if (next_operand == operands.end())
      break;
    out << format_constant(make_type(*(next_operand++), double_type2()));
    break;

  case 's':
  {
    if (next_operand == operands.end())
      break;
    const expr2tc &op = *(next_operand++);
    const expr2tc symbol2 = get_base_object(op);
    exprt char_array = migrate_expr_back(symbol2);
    if (char_array.id() == "string-constant")
      out << char_array.value().as_string();
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
    if (next_operand == operands.end())
      break;
    out << format_constant(make_type(*(next_operand++), char_type2()));
    break;

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
    const unsigned hex_chars = (config.ansi_c.pointer_width() + 3) / 4;
    const std::string placeholder = "0x" + std::string(hex_chars, '0');
    out << pad_int(placeholder, format_constant.min_width, false);
    break;
  }

  default:
    out << '%' << ch;
  }
}

void printf_formattert::process_char(std::ostream &out)
{
  char ch = next();

  if (ch == '%')
    process_format(out);
  else
    out << ch;
}
