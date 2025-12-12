#include <ctype.h>
#include <limits.h>

// Python character isalpha - handles ASCII letters only in a single-byte context.
_Bool __python_char_isalpha(int c)
{
__ESBMC_HIDE:;
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// Python string isalpha - handles ASCII and common two-byte UTF-8 Latin letters.
_Bool __python_str_isalpha(const char *s)
{
__ESBMC_HIDE:;
  if (!s || !*s)
    return 0;

  while (*s)
  {
    unsigned char c = (unsigned char)*s;

    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
    {
      s++;
      continue;
    }

    if (c <= 0x7F)
      return 0;

    if (c >= 0xC2 && c <= 0xDF)
    {
      unsigned char next = (unsigned char)*(s + 1);
      if (next >= 0x80 && next <= 0xBF)
      {
        s += 2;
        continue;
      }
    }

    return 0;
  }

  return 1;
}

// Python character isdigit - checks if a single character is a digit
_Bool __python_char_isdigit(int c)
{
__ESBMC_HIDE:;
  return (c >= '0' && c <= '9');
}

_Bool __python_str_isdigit(const char *s)
{
__ESBMC_HIDE:;
  if (!s || !*s)
    return 0;

  while (*s)
  {
    if (!isdigit((unsigned char)*s))
      return 0;
    s++;
  }
  return 1;
}

// Python string isspace: checks if all characters are whitespace
_Bool __python_str_isspace(const char *s)
{
__ESBMC_HIDE:;
  if (!s || !*s)
    return 0;

  while (*s)
  {
    unsigned char c = (unsigned char)*s;

    if (!(c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
          c == '\r'))
      return 0;

    s++;
  }

  return 1;
}

// Python string lstrip: removes leading whitespace characters
const char *__python_str_lstrip(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return s;

  while (*s && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\v' ||
                *s == '\f' || *s == '\r'))
  {
    s++;
  }

  return s;
}

// Python character islower - checks if a single character is lowercase
_Bool __python_char_islower(int c)
{
__ESBMC_HIDE:;
  return (c >= 'a' && c <= 'z');
}

// Python string islower - checks if all cased characters are lowercase
_Bool __python_str_islower(const char *s)
{
__ESBMC_HIDE:;
  if (!s || !*s)
    return 0;

  _Bool has_cased = 0;

  while (*s)
  {
    unsigned char c = (unsigned char)*s;

    if (c >= 'A' && c <= 'Z')
      return 0;

    if (c >= 'a' && c <= 'z')
      has_cased = 1;

    if (c >= 0xC2 && c <= 0xDF)
    {
      unsigned char next = (unsigned char)*(s + 1);
      if (next >= 0x80 && next <= 0xBF)
      {
        has_cased = 1;
        s += 2;
        continue;
      }
    }

    s++;
  }

  return has_cased;
}

// Python character lower - converts a single character to lowercase
int __python_char_lower(int c)
{
__ESBMC_HIDE:;
  if (c >= 'A' && c <= 'Z')
    return c + ('a' - 'A');
  return c;
}

// Python string lower - converts all characters to lowercase
char *__python_str_lower(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return (char *)s;

  char *buffer = __ESBMC_alloca(256);

  int i = 0;
  while (i < 255 && s[i])
  {
    if (s[i] >= 'A' && s[i] <= 'Z')
      buffer[i] = s[i] + ('a' - 'A');
    else
      buffer[i] = s[i];
    i++;
  }

  if (s[i] != '\0')
  {
    __ESBMC_assert(0, "String too long for lower() - exceeds 255 characters");
  }

  buffer[i] = '\0';

  return buffer;
}

int __python_str_find(const char *s1, const char *s2)
{
__ESBMC_HIDE:;
  // str.find('') = 0
  if (s2[0] == '\0')
    return 0;

  int len_s = strlen(s1);
  int len_sub = strlen(s2);

  int i = 0;
  while (i <= len_s - len_sub)
  {
    if (strncmp(s1 + i, s2, len_sub) == 0)
      return i;
    i++;
  }

  return -1;
}

// Python int() builtin - converts string to integer
int __python_int(const char *s, int base)
{
__ESBMC_HIDE:;
  if (!s)
    return 0;

  if (base != 0 && (base < 2 || base > 36))
  {
    __ESBMC_assert(0, "int() base must be >= 2 and <= 36, or 0");
    return 0;
  }

  while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\v' || *s == '\f' ||
         *s == '\r')
    s++;

  if (!*s)
  {
    __ESBMC_assert(0, "invalid literal for int() with empty string");
    return 0;
  }

  int sign = 1;
  if (*s == '-')
  {
    sign = -1;
    s++;
  }
  else if (*s == '+')
  {
    s++;
  }

  if (base == 0)
  {
    if (*s == '0')
    {
      if (*(s + 1) == 'x' || *(s + 1) == 'X')
      {
        base = 16;
        s += 2;
      }
      else if (*(s + 1) == 'b' || *(s + 1) == 'B')
      {
        base = 2;
        s += 2;
      }
      else if (*(s + 1) == 'o' || *(s + 1) == 'O')
      {
        base = 8;
        s += 2;
      }
      else
      {
        base = 10;
      }
    }
    else
    {
      base = 10;
    }
  }
  else if (base == 16 && *s == '0' && (*(s + 1) == 'x' || *(s + 1) == 'X'))
  {
    s += 2;
  }
  else if (base == 2 && *s == '0' && (*(s + 1) == 'b' || *(s + 1) == 'B'))
  {
    s += 2;
  }
  else if (base == 8 && *s == '0' && (*(s + 1) == 'o' || *(s + 1) == 'O'))
  {
    s += 2;
  }

  if (!*s)
  {
    __ESBMC_assert(0, "invalid literal for int() - no digits");
    return 0;
  }

  int result = 0;
  _Bool found_digit = 0;

  while (*s)
  {
    int digit_value = -1;
    unsigned char c = (unsigned char)*s;

    if (c >= '0' && c <= '9')
    {
      digit_value = c - '0';
    }
    else if (c >= 'a' && c <= 'z')
    {
      digit_value = c - 'a' + 10;
    }
    else if (c >= 'A' && c <= 'Z')
    {
      digit_value = c - 'A' + 10;
    }
    else if (
      c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r')
    {
      s++;
      continue;
    }
    else
    {
      __ESBMC_assert(0, "invalid literal for int() - invalid character");
      return 0;
    }

    if (digit_value >= base)
    {
      if (found_digit)
      {
        break;
      }
      __ESBMC_assert(
        0, "invalid literal for int() - digit out of range for base");
      return 0;
    }

    found_digit = 1;

    if (result > (INT_MAX / base))
    {
      __ESBMC_assert(0, "int() conversion overflow");
      return 0;
    }

    result = result * base + digit_value;
    s++;
  }

  while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\v' || *s == '\f' ||
         *s == '\r')
    s++;

  if (*s != '\0')
  {
    __ESBMC_assert(0, "invalid literal for int() - trailing characters");
    return 0;
  }

  if (!found_digit)
  {
    __ESBMC_assert(0, "invalid literal for int() - no valid digits");
    return 0;
  }

  return sign * result;
}

// Python chr() builtin - converts Unicode code point to string
char *__python_chr(int codepoint)
{
__ESBMC_HIDE:;
  if (codepoint < 0 || codepoint > 0x10FFFF)
  {
    __ESBMC_assert(0, "chr() arg not in range(0x110000)");
    return (char *)0;
  }

  if (codepoint >= 0xD800 && codepoint <= 0xDFFF)
  {
    __ESBMC_assert(0, "chr() arg is a surrogate code point");
    return (char *)0;
  }

  char *buffer = __ESBMC_alloca(5);

  if (codepoint <= 0x7F)
  {
    buffer[0] = (char)codepoint;
    buffer[1] = '\0';
    return buffer;
  }

  if (codepoint <= 0x7FF)
  {
    buffer[0] = (char)(0xC0 | (codepoint >> 6));
    buffer[1] = (char)(0x80 | (codepoint & 0x3F));
    buffer[2] = '\0';
    return buffer;
  }

  if (codepoint <= 0xFFFF)
  {
    buffer[0] = (char)(0xE0 | (codepoint >> 12));
    buffer[1] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
    buffer[2] = (char)(0x80 | (codepoint & 0x3F));
    buffer[3] = '\0';
    return buffer;
  }

  buffer[0] = (char)(0xF0 | (codepoint >> 18));
  buffer[1] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
  buffer[2] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
  buffer[3] = (char)(0x80 | (codepoint & 0x3F));
  buffer[4] = '\0';
  return buffer;
}

// Python string concatenation - combines two strings
char *__python_str_concat(const char *s1, const char *s2)
{
__ESBMC_HIDE:;
  if (!s1 && !s2)
    return (char *)0;

  if (!s1)
    s1 = "";
  if (!s2)
    s2 = "";

  char *buffer = __ESBMC_alloca(512);

  int i = 0;
  int pos = 0;

  // Copy first string
  while (pos < 511 && s1[i])
  {
    buffer[pos] = s1[i];
    pos++;
    i++;
  }

  if (s1[i] != '\0')
  {
    __ESBMC_assert(0, "String concatenation overflow - first string too long");
    buffer[511] = '\0';
    return buffer;
  }

  // Copy second string
  i = 0;
  while (pos < 511 && s2[i])
  {
    buffer[pos] = s2[i];
    pos++;
    i++;
  }

  if (s2[i] != '\0')
  {
    __ESBMC_assert(
      0, "String concatenation overflow - result exceeds 511 characters");
  }

  buffer[pos] = '\0';
  return buffer;
}
