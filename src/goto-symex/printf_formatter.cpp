/*******************************************************************\

Module: printf Formatting

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-symex/printf_formatter.h>
#include <sstream>
#include <util/c_types.h>
#include <util/irep2_utils.h>
#include <util/format_constant.h>
#include <util/type_byte_size.h>

const expr2tc printf_formattert::make_type(
  const expr2tc &src,
  const type2tc &dest)
{
  if(src->type == dest)
    return src;

  typecast2tc tmp(dest, src);
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
    while(!eol())
      process_char(out);
  }

  catch(eol_exception)
  {
  }
}

std::string printf_formattert::as_string()
{
  std::ostringstream stream;
  print(stream);
  return stream.str();
}

void printf_formattert::process_format(std::ostream &out)
{
  expr2tc tmp;
  format_constantt format_constant;

  format_constant.precision = 6;
  format_constant.min_width = 0;
  format_constant.zero_padding = false;

  char ch = next();

  if(ch == '0') // leading zeros
  {
    format_constant.zero_padding = true;
    ch = next();
  }

  while(isdigit(ch)) // width
  {
    format_constant.min_width *= 10;
    format_constant.min_width += ch - '0';
    ch = next();
  }

  if(ch == '.') // precision
  {
    format_constant.precision = 0;
    ch = next();

    while(isdigit(ch))
    {
      format_constant.precision *= 10;
      format_constant.precision += ch - '0';
      ch = next();
    }
  }

  switch (ch)
  {
    case '%':
      out << ch;
      break;

    case 'e':
    case 'E':
      format_constant.style = format_spect::stylet::SCIENTIFIC;
      if(next_operand == operands.end())
        break;
      out << format_constant(make_type(*(next_operand++), double_type2()));
      break;

    case 'f':
    case 'F':
      format_constant.style = format_spect::stylet::DECIMAL;
      if(next_operand == operands.end())
        break;
      out << format_constant(make_type(*(next_operand++), double_type2()));
      break;

    case 'g':
    case 'G':
      format_constant.style = format_spect::stylet::AUTOMATIC;
      if(format_constant.precision == 0)
        format_constant.precision = 1;
      if(next_operand == operands.end())
        break;
      out << format_constant(make_type(*(next_operand++), double_type2()));
      break;

    case 's':
    {
      if(next_operand == operands.end())
        break;

      // this is the address of a string
      const expr2tc &op = *(next_operand++);
      if(is_address_of2t(op) && is_string_type(to_address_of2t(op).ptr_obj))
        out << format_constant(to_address_of2t(op).ptr_obj);
    }
      break;

    case 'd':
      if(next_operand == operands.end())
        break;
      out << format_constant(make_type(*(next_operand++), int_type2()));
      break;

    case 'D':
      if(next_operand == operands.end())
        break;
      out << format_constant(make_type(*(next_operand++), long_int_type2()));
      break;

    case 'u':
      if(next_operand == operands.end())
        break;
      out << format_constant(make_type(*(next_operand++), uint_type2()));
      break;

    case 'U':
      if(next_operand == operands.end())
        break;
      out << format_constant(make_type(*(next_operand++), long_uint_type2()));
      break;

    default:
      out << '%' << ch;
  }
}

void printf_formattert::process_char(std::ostream &out)
{
  char ch = next();

  if(ch == '%')
    process_format(out);
  else
    out << ch;
}
