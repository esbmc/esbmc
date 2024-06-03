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

#if __APPLE__ && __MACH__
/* Apple enjoys their users suffering to write platform-independent code, that's
 * why they chose to *not* provide the c(16|32)rtomb functions. */

static_assert(sizeof(wchar_t) == sizeof(char32_t));
static_assert(sizeof(mbstate_t) >= sizeof(uint32_t));

#  define UNI_PLANE_SZ 0x10000
#  define SURR_VAL_BITS 10
#  define SURR_VAL_MASK ((1 << SURR_VAL_BITS) - 1)
#  define SURR_HI_MASK 0xd800
#  define SURR_LO_MASK 0xdc00
#  define IS_SURR(c) (((c) & ~0x7ff) == 0xd800)
#  define IS_SURR_HI(c) (((c) & ~SURR_VAL_MASK) == SURR_HI_MASK)
#  define IS_SURR_LO(c) (((c) & ~SURR_VAL_MASK) == SURR_LO_MASK)
#  define SURR_COMBINE(hi, lo)                                                 \
    (((char32_t)(((hi)&SURR_VAL_MASK) << SURR_VAL_BITS) + UNI_PLANE_SZ) |      \
     (char32_t)((lo)&SURR_VAL_MASK))

/* Define the missing functions, statically, so once they choose to implement
 * them in their libc, we'll be notified and have to see about the version... */
static size_t c32rtomb(char *buf, char32_t c, mbstate_t *ps)
{
  /* Assumption: wchar_t is UTF-32, same as char32_t */
  return wcrtomb(buf, static_cast<wchar_t>(c), ps);
}

static size_t c16rtomb(char *buf, char16_t c, mbstate_t *ps)
{
  /* Convert from UTF-16 to UTF-32 and use c32rtomb() */

  /* Initial state     <-> low surrogate not allowed
   * Not initial state <-> low surrogate expected */
  bool init = mbsinit(ps);
  if (!init == !IS_SURR_LO(c))
  {
    errno = EILSEQ;
    return -1;
  }

  /* We need to store 2 values in the state: the high surrogate (comes first)
   * and a marker (in case the high surrogate was 0) */
  uint16_t *s = reinterpret_cast<uint16_t *>(ps);
  if (IS_SURR_HI(c))
  {
    assert(init);
    s[0] = c;
    s[1] = 1;
    return 0;
  }

  char32_t d = c;
  if (IS_SURR_LO(c))
  {
    assert(s[1] == 1);
    d = SURR_COMBINE(s[0], c);
    memset(ps, 0, sizeof(*ps));
  }
  return c32rtomb(buf, d, ps);
}
#endif

namespace
{
/* RAII-like type to temporarily switch the locale */
struct switch_locale
{
  std::string orig;
  int cat;

  switch_locale(int cat, const char *loc)
    : orig(setlocale(cat, nullptr)), cat(cat)
  {
    const char *n = setlocale(cat, loc);
    if (!n)
      throw fmt::format("locale '{}' not found", loc);
    log_debug(
      "convert_mb", "switching from locale '{}' to '{}' -> '{}'", orig, loc, n);
  }

  ~switch_locale() noexcept
  {
    const char *n = setlocale(cat, orig.c_str()); // restore locale
    assert(n);
    assert(n == orig);
  }
};

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
      le(config.ansi_c.endianess == configt::ansi_ct::IS_LITTLE_ENDIAN)
  {
    assert(v.length() % w == 0);
    if (wide && sizeof(wchar_t) * 8 != config.ansi_c.wchar_t_width)
      throw string_constantt::mb_conversion_error(fmt::format(
        "error interpreting {} string literal: host and target wchar_t widths "
        "differ: {} != {}",
        desc(),
        sizeof(wchar_t) * 8,
        config.ansi_c.wchar_t_width));
    if (config.ansi_c.endianess == configt::ansi_ct::NO_ENDIANESS)
      throw string_constantt::mb_conversion_error(fmt::format(
        "impossible to interpret {} string literal without endianness",
        desc()));

    memset(&ps, 0, sizeof(ps));

    try
    {
      /* need to set the locale for (wc|c16|c32)rtomb() to work outside of
       * ASCII; it'll be restored by the destructor at the end of this block */
      switch_locale nloc(LC_CTYPE, config.ansi_c.locale_name.c_str());
      size_t n = v.length() / w; // number of code units

      std::vector<char> buffer(MB_CUR_MAX);
      char *buf = buffer.data();
      for (size_t i = 0; i < n; i++)
      {
        uint32_t c = decode(i);
        size_t r = encode(buf, c);
        if (r == (size_t)-1)
          throw string_constantt::mb_conversion_error(fmt::format(
            "error interpreting {} string literal at {}: {}",
            desc(),
            i,
            strerror(errno)));
        result.insert(result.end(), buf, buf + r);
      }
    }
    catch (const std::string &ex)
    {
      throw string_constantt::mb_conversion_error(
        fmt::format("error interpreting {} string literal: {}", desc(), ex));
    }

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
    assert(c || mbsinit(&ps));
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
  bool le = config.ansi_c.endianess == configt::ansi_ct::IS_LITTLE_ENDIAN;
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
    uint32_t c = *p++ & (0xff >> n);
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
  int elem_width = bv_width(type().subtype());
  irep_idt kind = get("kind");
  bool is_wide = kind == k_wide;
  bool is_default = kind == k_default;
  switch (elem_width)
  {
  case 8:
    assert(!is_wide);
    if (is_default)
      return value();
    return convert_utf8(value().as_string());
  case 16:
  case 32:
    assert(!is_default);
    return convert_mb(value().as_string(), elem_width / 8, is_wide).result;
  }
  log_error("illegal character width {} of string literal", elem_width);
  abort();
}
