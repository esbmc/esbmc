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

const irep_idt string_constantt::k_default = "default";
const irep_idt string_constantt::k_wide = "wide";
const irep_idt string_constantt::k_unicode = "unicode";

string_constantt::string_constantt(const irep_idt &value)
  : string_constantt(
      value,
      array_typet(char_type(), constant_exprt(value.size() + 1, size_type())),
      k_default)
{
}

string_constantt::string_constantt(
  const irep_idt &value,
  const typet &type,
  const irep_idt &kind)
  : exprt("string-constant", type)
{
  exprt::value(value);
  set("kind", kind);
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
    std::string orig_loc = setlocale(LC_CTYPE, nullptr);
    if (!setlocale(LC_CTYPE, config.ansi_c.locale_name.c_str()))
      throw string_constantt::mb_conversion_error(fmt::format(
        "error interpreting {} string literal: locale '{}' not found",
        desc(),
        config.ansi_c.locale_name));

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

    char *loc = setlocale(LC_CTYPE, orig_loc.c_str()); // restore locale
    assert(loc == orig_loc);

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

/* TODO: once we switch to C++20, unify this with convert_mb() via c8rtomb() */
static std::string convert_utf8(const std::string &v)
{
  /* First convert the entire string from UTF-8 to UTF-32 in the target's
   * endianness, then convert that to the host. This is quite slow... */
  std::string tmp;
  const uint8_t *p0 = reinterpret_cast<const uint8_t *>(v.data());
  const uint8_t *p = p0;
  const uint8_t *e = p0 + v.length();
  bool le =
    config.ansi_c.endianess == configt::ansi_ct::endianesst::IS_LITTLE_ENDIAN;
  while (p != e)
  {
    const uint8_t *p1 = p;
    int n = *p < 0x80   ? 1 // 0xxxxxxx
            : *p < 0xc0 ? 0
            : *p < 0xe0 ? 2 // 110xxxxx 10xxxxxx
            : *p < 0xf0 ? 3 // 1110xxxx 10xxxxxx 10xxxxxx
            : *p < 0xf8 ? 4 // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                        : 0;
    if (!n || e - p < n)
      throw string_constantt::mb_conversion_error(fmt::format(
        "error interpreting UTF-8 string literal: invalid sequence at {}",
        p1 - p0));
    uint32_t c = *p++;
    switch (n)
    {
    case 2:
      c &= ~0xc0U;
      break;
    case 3:
      c &= ~0xe0U;
      break;
    case 4:
      c &= ~0xf0U;
      break;
    }
    for (int i = 1; i < n; i++, p++)
    {
      if (*p < 0x80 || *p >= 0xc0)
        throw string_constantt::mb_conversion_error(fmt::format(
          "error interpreting UTF-8 string literal: invalid sequence at {}",
          p1 - p0));
      c = (c << 6) | (*p & ~0xc0U);
    }
    char buf[4];
    for (int i = 0; i < 4; i++)
      buf[i] = (c >> 8 * (le ? i : 4 - 1 - i)) & 0xff;
    tmp.append(buf, 4);
  }
  return convert_mb(tmp, 4, false).result;
}

irep_idt string_constantt::mb_value() const
{
  int elem_width = atoi(type().subtype().width().c_str());
  irep_idt kind = get("kind");
  bool is_wide = kind == k_wide;
  switch (elem_width)
  {
  case 8:
    assert(!is_wide);
    if (kind == k_default)
      return value();
    return convert_utf8(value().as_string());
  case 16:
  case 32:
    return convert_mb(value().as_string(), elem_width / 8, is_wide).result;
  }
  log_error("illegal character width {} of string literal", elem_width);
  abort();
}
