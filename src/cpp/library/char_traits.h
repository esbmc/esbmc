// std::char_traits, shared by the string, string_view and stream OMs.
//
// Lifted out of <string> unchanged. It lived there, but <string> includes
// <string_view>, so <string_view> could not reach it, and the stream OMs and
// <iterator> only ever forward-declared it to name it as a default template
// argument. Any program that touched a member from those headers -- fmt and
// boost lexical_cast do -- failed to parse. See #5868.
//
// The `Traits = char_traits<CharT>` defaults stay with the templates that
// declare them; repeating one here would be a second default for the same
// parameter.
#pragma once

#include <cstddef> /* size_t, ptrdiff_t */
#include <cstdio>  /* EOF */
#include <cstring> /* memcpy, memmove */

#include "OM_compiler_defs.h"

namespace std
{
template <class charT>
struct char_traits
{
  typedef charT char_type;
  typedef int int_type;
  typedef std::ptrdiff_t off_type;
  typedef std::size_t pos_type;

  static void assign(char_type &c1, const char_type &c2) OM_NOEXCEPT
  {
    c1 = c2;
  }

  static char_type *assign(char_type *s, std::size_t n, char_type a)
  {
    for (std::size_t i = 0; i < n; ++i)
      s[i] = a;
    return s;
  }

  static bool eq(char_type c1, char_type c2) OM_NOEXCEPT
  {
    return c1 == c2;
  }

  static bool lt(char_type c1, char_type c2) OM_NOEXCEPT
  {
    return c1 < c2;
  }

  static int compare(const char_type *s1, const char_type *s2, std::size_t n)
  {
    for (std::size_t i = 0; i < n; ++i)
    {
      if (lt(s1[i], s2[i]))
        return -1;
      if (lt(s2[i], s1[i]))
        return 1;
    }
    return 0;
  }

  static std::size_t length(const char_type *s)
  {
    std::size_t i = 0;
    while (!eq(s[i], char_type(0)))
      ++i;
    return i;
  }

  static const char_type *
  find(const char_type *s, std::size_t n, const char_type &a)
  {
    for (std::size_t i = 0; i < n; ++i)
      if (eq(s[i], a))
        return s + i;
    return OM_NULLPTR;
  }

  static char_type *move(char_type *dest, const char_type *src, std::size_t n)
  {
    return static_cast<char_type *>(memmove(dest, src, n * sizeof(charT)));
  }

  static char_type *copy(char_type *dest, const char_type *src, std::size_t n)
  {
    return static_cast<char_type *>(memcpy(dest, src, n * sizeof(charT)));
  }

  static int_type eof() OM_NOEXCEPT
  {
    return EOF;
  }

  static int_type to_int_type(char_type c) OM_NOEXCEPT
  {
    return static_cast<unsigned char>(c);
  }

  static char_type to_char_type(int_type c) OM_NOEXCEPT
  {
    return static_cast<char_type>(c);
  }

  static bool eq_int_type(int_type c1, int_type c2) OM_NOEXCEPT
  {
    return c1 == c2;
  }

  static int_type not_eof(int_type c) OM_NOEXCEPT
  {
    return (c == eof()) ? 0 : c;
  }
};

// [char.traits.specializations.char]: "The two-argument members eq and lt are
// defined identically to the built-in operators == and < for type unsigned
// char." The generic template compares as char_type, which orders anything
// above 0x7f wrongly wherever char is signed -- and compare(), hence
// std::string's ordering, is built on lt. See #7048.
template <>
inline bool char_traits<char>::eq(char c1, char c2) OM_NOEXCEPT
{
  return static_cast<unsigned char>(c1) == static_cast<unsigned char>(c2);
}

template <>
inline bool char_traits<char>::lt(char c1, char c2) OM_NOEXCEPT
{
  return static_cast<unsigned char>(c1) < static_cast<unsigned char>(c2);
}
} // namespace std
