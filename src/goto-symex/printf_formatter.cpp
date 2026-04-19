#include <goto-symex/printf_formatter.h>
#include <sstream>
#include <util/c_types.h>
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

static std::string pad_int(
  const std::string &s,
  unsigned min_width,
  bool zero_padding)
{
  if (s.length() >= min_width)
    return s;
  return std::string(min_width - s.length(), zero_padding ? '0' : ' ') + s;
}

static std::string format_hex(
  unsigned long val,
  bool uppercase,
  unsigned min_width,
  bool zero_padding)
{
  std::ostringstream oss;
  if (uppercase)
    oss << std::uppercase;
  oss << std::hex << val;
  return pad_int(oss.str(), min_width, zero_padding);
}

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

  // Skip length modifiers (h, l, ll, L, z, j, t)
  while (
    ch == 'h' || ch == 'l' || ch == 'L' || ch == 'z' || ch == 'j' ||
    ch == 't')
    ch = next();

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
  {
    if (next_operand == operands.end())
      break;
    std::string s =
      format_constant(make_type(*(next_operand++), int_type2()));
    out << pad_int(s, format_constant.min_width, format_constant.zero_padding);
    break;
  }

  case 'D':
  {
    if (next_operand == operands.end())
      break;
    std::string s =
      format_constant(make_type(*(next_operand++), long_int_type2()));
    out << pad_int(s, format_constant.min_width, format_constant.zero_padding);
    break;
  }

  case 'u':
  {
    if (next_operand == operands.end())
      break;
    std::string s =
      format_constant(make_type(*(next_operand++), uint_type2()));
    out << pad_int(s, format_constant.min_width, format_constant.zero_padding);
    break;
  }

  case 'U':
  {
    if (next_operand == operands.end())
      break;
    std::string s =
      format_constant(make_type(*(next_operand++), long_uint_type2()));
    out << pad_int(s, format_constant.min_width, format_constant.zero_padding);
    break;
  }

  case 'c':
    if (next_operand == operands.end())
      break;
    out << format_constant(make_type(*(next_operand++), char_type2()));
    break;

  case 'x':
  case 'X':
  {
    if (next_operand == operands.end())
      break;
    const expr2tc casted = make_type(*(next_operand++), uint_type2());
    unsigned long val =
      is_constant_int2t(casted) ? to_constant_int2t(casted).as_ulong() : 0;
    out << format_hex(
      val,
      ch == 'X',
      format_constant.min_width,
      format_constant.zero_padding);
    break;
  }

  case 'o':
  {
    if (next_operand == operands.end())
      break;
    const expr2tc casted = make_type(*(next_operand++), uint_type2());
    unsigned long val =
      is_constant_int2t(casted) ? to_constant_int2t(casted).as_ulong() : 0;
    std::ostringstream oss;
    oss << std::oct << val;
    out << pad_int(
      oss.str(), format_constant.min_width, format_constant.zero_padding);
    break;
  }

  case 'p':
    // Consume the pointer argument; output a placeholder
    if (next_operand != operands.end())
      ++next_operand;
    out << pad_int("0", format_constant.min_width, false);
    break;

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
