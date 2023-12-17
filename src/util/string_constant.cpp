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
  : string_constantt(
      value,
      array_typet(char_type(), constant_exprt(value.size() + 1, size_type())),
      false)
{
}

string_constantt::string_constantt(
  const irep_idt &value,
  const typet &type,
  bool is_wide)
  : exprt("string-constant", type)
{
  exprt::value(value);
  set("wide", is_wide);
}

namespace
{
struct convert_mb
{
  const std::string &v;
  int w;
  bool wide;
  bool le;

  mbstate_t ps;
  std::string result;

  convert_mb(const std::string &v, int w, bool wide)
    : v(v),
      w(w),
      wide(wide),
      le(
        config.ansi_c.endianess ==
        configt::ansi_ct::endianesst::IS_LITTLE_ENDIAN)
  {
    assert(v.length() % w == 0);
    if (config.ansi_c.endianess == configt::ansi_ct::endianesst::NO_ENDIANESS)
      throw string_constantt::mb_conversion_error(fmt::format(
        "impossible to interpret {} string literal without endianness",
        desc()));

    memset(&ps, 0, sizeof(ps));
    size_t n = v.length() / w; // number of code units
    size_t i;

    /* need to set the locale for c*rtomb() to work; we'll restore it later */
    char *loc = setlocale(LC_CTYPE, config.ansi_c.locale_name.c_str());
    assert(loc);
    std::vector<char> buffer(MB_CUR_MAX);
    char *buf = buffer.data();
    for (i = 0; i < n; i++)
    {
      uint32_t c = decode(i);
      size_t r = encode(buf, c);
      if (r == (size_t)-1)
        break;
      result.insert(result.end(), buf, buf + r);
    }
    setlocale(LC_CTYPE, loc); // restore locale

    if (i < n)
      throw string_constantt::mb_conversion_error(fmt::format(
        "error interpreting {} string literal at {}: {}",
        desc(),
        i,
        strerror(errno)));
    if (!mbsinit(&ps))
      throw string_constantt::mb_conversion_error(fmt::format(
        "error interpreting {} string literal: terminates with "
        "incomplete sequence",
        desc()));
  }

  std::string desc() const
  {
    return wide ? std::string("wchar_t") : fmt::format("char{}_t", 8 * w);
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

  size_t encode(char *buf, uint32_t c)
  {
    return wide     ? wcrtomb(buf, c, &ps)
           : w == 2 ? c16rtomb(buf, c, &ps)
                    : c32rtomb(buf, c, &ps);
  }
};

} // namespace

irep_idt string_constantt::mb_value() const
{
  /* Assume all strings are Unicode, in particular char * and wchar_t * are also
   * Unicode (TODO: which isn't true in China: they use GB18030). */
  int elem_width = atoi(type().subtype().width().c_str());
  bool is_wide = get_bool("wide");
  switch (elem_width)
  {
  case 8:
    assert(!is_wide);
    return value();

  case 16:
  case 32:
    return convert_mb(value().as_string(), elem_width / 8, is_wide).result;
  }
  log_error("illegal character width {} of string literal", elem_width);
  abort();
}
