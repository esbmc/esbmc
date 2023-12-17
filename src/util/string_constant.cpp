#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/config.h>
#include <util/c_types.h>
#include <util/message.h>
#include <util/std_expr.h>
#include <util/std_types.h>
#include <util/string_constant.h>

#include <cuchar>
#include <cwchar>

string_constantt::string_constantt(const irep_idt &value)
  : string_constantt(value, array_typet(char_type()))
{
}

string_constantt::string_constantt(const irep_idt &value, const typet &type)
  : exprt("string-constant", type)
{
  set_value(value);
}

void string_constantt::set_value(const irep_idt &value)
{
  exprt::value(value);
}

namespace
{
struct convert_mb
{
  const std::string &v;
  int w;
  bool le;

  std::vector<char> res;

  convert_mb(const std::string &v, int w)
    : v(v),
      w(w),
      le(
        config.ansi_c.endianess ==
        configt::ansi_ct::endianesst::IS_LITTLE_ENDIAN)
  {
    assert(v.length() % w == 0);
    if (config.ansi_c.endianess == configt::ansi_ct::endianesst::NO_ENDIANESS)
      throw string_constantt::mb_conversion_error(fmt::format(
        "impossible to interpret char{}_t string literal without endianness",
        8 * w));

    char buffer[MB_CUR_MAX];
    mbstate_t ps;
    memset(&ps, 0, sizeof(ps));
    size_t n = v.length() / w; // number of code units
    size_t i;

    /* need to set the locale for c*rtomb() to work; we'll restore it later */
    char *loc = setlocale(LC_CTYPE, config.ansi_c.locale_name.c_str());
    assert(loc);
    for (i = 0; i < n; i++)
    {
      uint32_t c = decode(i);
      size_t r = w == 2 ? c16rtomb(buffer, c, &ps) : c32rtomb(buffer, c, &ps);
      if (r == (size_t)-1)
        break;
      res.insert(res.end(), buffer, buffer + r);
    }
    setlocale(LC_CTYPE, loc); // restore locale

    if (i < n)
      throw string_constantt::mb_conversion_error(fmt::format(
        "error interpreting char{}_t string literal at {}: {}",
        8 * w,
        i,
        strerror(errno)));
    if (!mbsinit(&ps))
      throw string_constantt::mb_conversion_error(fmt::format(
        "error interpreting char{}_t string literal: terminates with "
        "incomplete sequence",
        8 * w));
  }

  uint32_t decode(size_t k) const
  {
    const char *p = v.data() + k * w;
    uint32_t r = 0;
    for (int i = 0; i < w; i++)
    {
      uint8_t c = p[i];
      r |= (uint32_t)c << 8 * (le ? i : w - 1 - i);
    }
    return r;
  }

  std::string result() const
  {
    return std::string(res.begin(), res.end());
  }
};

} // namespace

irep_idt string_constantt::mb_value() const
{
  /* Assume wchar_t is either char16_t or char32_t. */
  int elem_width = atoi(type().subtype().width().c_str());
  switch (elem_width) {
  case 8:
    return get_value();

  case 16:
  case 32:
    return convert_mb(get_value().as_string(), elem_width / 8).result();
  }
  log_error("illegal character width {} of string literal", elem_width);
  abort();
}
