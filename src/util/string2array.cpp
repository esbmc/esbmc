#include <util/arith_tools.h>
#include <util/config.h>
#include <util/std_types.h>
#include <util/string2array.h>

void string2array(const exprt &src, exprt &dest)
{
  const std::string &str = src.value().as_string();
  unsigned actual_size = str.size();

  BigInt str_size;
  to_integer(to_array_type(src.type()).size(), str_size);
  uint64_t string_size = str_size.to_uint64();

  const typet &char_type = src.type().subtype();
  bool char_is_unsigned = char_type.is_unsignedbv();

  exprt size("constant", typet("signedbv"));
  size.type().width(config.ansi_c.int_width);
  size.value(integer2binary(string_size, config.ansi_c.int_width));

  dest = exprt("constant", typet("array"));
  dest.type().subtype() = char_type;
  dest.type().size(size);

  dest.operands().resize(string_size);

  exprt::operandst::iterator it = dest.operands().begin();
  for(unsigned i = 0; i < string_size; i++, it++)
  {
    int ch = i >= actual_size ? 0 : str[i];

    if(char_is_unsigned)
      ch = (unsigned char)ch;

    exprt &op = *it;

    op = from_integer(ch, char_type);

    if(ch >= 32 && ch <= 126)
    {
      char ch_str[2];
      ch_str[0] = ch;
      ch_str[1] = 0;

      op.cformat("'" + std::string(ch_str) + "'");
    }
  }
}
