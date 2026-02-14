#include <ctype.h>
#include <stdbool.h>
#include <limits.h>
#include <stddef.h>
#include <string.h>
#include "python_types.h"

#define ESBMC_PY_STR_TYPE_ID ((size_t)0x826e83195d0d60f0ULL)

// List helpers (defined in list.c)
extern PyListObject *__ESBMC_list_create(void);
extern bool __ESBMC_list_push(
  PyListObject *l,
  const void *value,
  size_t type_id,
  size_t type_size);
extern size_t __ESBMC_list_size(const PyListObject *l);
extern PyObject *__ESBMC_list_at(PyListObject *l, size_t index);

// Python character isalpha - handles ASCII letters only in a single-byte context.
_Bool __python_char_isalpha(int c)
{
__ESBMC_HIDE:;
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

size_t __python_strnlen_bounded(const char *s, size_t max_len)
{
__ESBMC_HIDE:;
  size_t i = 0;
  while (i < max_len)
  {
    if (s[i] == '\0')
      return i;
    i++;
  }

  __ESBMC_assert(0, "string not null-terminated");
  return max_len;
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

static void __python_str_normalize_range(int *start, int *end, size_t len_s)
{
__ESBMC_HIDE:;
  int len_i = (int)len_s;

  if (*start == INT_MIN)
    *start = 0;

  if (*end == INT_MIN)
    *end = len_i;

  if (*start < 0)
    *start = *start + len_i;
  if (*end < 0)
    *end = *end + len_i;

  if (*start < 0)
    *start = 0;
  if (*start > len_i)
    *start = len_i;

  if (*end < 0)
    *end = 0;
  if (*end > len_i)
    *end = len_i;

  // Do not force end >= start; Python allows empty ranges when start > end.
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

// Python string rstrip: removes trailing whitespace characters
const char *__python_str_rstrip(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return s;

  const char *start = s;
  const char *end = start;

  while (*end)
  {
    end++;
  }

  while (end > start &&
         (*(end - 1) == ' ' || *(end - 1) == '\t' || *(end - 1) == '\n' ||
          *(end - 1) == '\v' || *(end - 1) == '\f' || *(end - 1) == '\r'))
  {
    end--;
  }

  size_t len = (size_t)(end - start);
  char *buffer = __ESBMC_alloca(len + 1);

  size_t i = 0;
  while (i < len)
  {
    buffer[i] = start[i];
    ++i;
  }

  buffer[len] = '\0';

  return buffer;
}

// Python string strip: removes leading and trailing whitespace characters
const char *__python_str_strip(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return s;

  while (*s && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\v' ||
                *s == '\f' || *s == '\r'))
  {
    s++;
  }

  const char *start = s;
  const char *end = start;

  while (*end)
  {
    end++;
  }

  while (end > start &&
         (*(end - 1) == ' ' || *(end - 1) == '\t' || *(end - 1) == '\n' ||
          *(end - 1) == '\v' || *(end - 1) == '\f' || *(end - 1) == '\r'))
  {
    end--;
  }

  size_t len = (size_t)(end - start);
  char *buffer = __ESBMC_alloca(len + 1);

  size_t i = 0;
  while (i < len)
  {
    buffer[i] = start[i];
    ++i;
  }

  buffer[len] = '\0';

  return buffer;
}

// Python string strip with custom chars - removes chars from both ends
const char *__python_str_strip_chars(const char *s, const char *chars)
{
__ESBMC_HIDE:;
  if (!s)
    return s;
  if (!chars || !*chars)
    return __python_str_strip(s);

  // Skip leading chars
  while (*s)
  {
    _Bool found = 0;
    const char *p = chars;
    while (*p && !found)
    {
      if (*p == *s)
        found = 1;
      p++;
    }
    if (!found)
      break;
    s++;
  }

  const char *start = s;
  const char *end = start;

  // Find the end
  while (*end)
  {
    end++;
  }

  // Skip trailing chars
  while (end > start)
  {
    _Bool found = 0;
    const char *p = chars;
    char ch = *(end - 1);
    while (*p && !found)
    {
      if (*p == ch)
        found = 1;
      p++;
    }
    if (!found)
      break;
    end--;
  }

  size_t len = (size_t)(end - start);
  char *buffer = __ESBMC_alloca(len + 1);

  size_t i = 0;
  while (i < len)
  {
    buffer[i] = start[i];
    ++i;
  }

  buffer[len] = '\0';

  return buffer;
}

// Python string lstrip with custom chars - removes chars from left
const char *__python_str_lstrip_chars(const char *s, const char *chars)
{
__ESBMC_HIDE:;
  if (!s)
    return s;
  if (!chars || !*chars)
    return __python_str_lstrip(s);

  // Skip leading chars
  while (*s)
  {
    _Bool found = 0;
    const char *p = chars;
    while (*p && !found)
    {
      if (*p == *s)
        found = 1;
      p++;
    }
    if (!found)
      break;
    s++;
  }

  const char *start = s;
  const char *end = start;

  // Find the end
  while (*end)
  {
    end++;
  }

  size_t len = (size_t)(end - start);
  char *buffer = __ESBMC_alloca(len + 1);

  size_t i = 0;
  while (i < len)
  {
    buffer[i] = start[i];
    ++i;
  }

  buffer[len] = '\0';

  return buffer;
}

// Python string rstrip with custom chars - removes chars from right
const char *__python_str_rstrip_chars(const char *s, const char *chars)
{
__ESBMC_HIDE:;
  if (!s)
    return s;
  if (!chars || !*chars)
    return __python_str_rstrip(s);

  const char *start = s;
  const char *end = start;

  // Find the end
  while (*end)
  {
    end++;
  }

  // Skip trailing chars
  while (end > start)
  {
    _Bool found = 0;
    const char *p = chars;
    char ch = *(end - 1);
    while (*p && !found)
    {
      if (*p == ch)
        found = 1;
      p++;
    }
    if (!found)
      break;
    end--;
  }

  size_t len = (size_t)(end - start);
  char *buffer = __ESBMC_alloca(len + 1);

  size_t i = 0;
  while (i < len)
  {
    buffer[i] = start[i];
    ++i;
  }

  buffer[len] = '\0';

  return buffer;
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

// Python character upper - converts a single character to uppercase
int __python_char_upper(int c)
{
__ESBMC_HIDE:;
  if (c >= 'a' && c <= 'z')
    return c - ('a' - 'A');
  return c;
}

// Python string lower - converts all characters to lowercase
char *__python_str_lower(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return (char *)s;

  char *buffer = __ESBMC_alloca(256);

  size_t i = 0;
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

// Python string upper - converts all characters to uppercase
char *__python_str_upper(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return (char *)s;

  char *buffer = __ESBMC_alloca(256);

  int i = 0;
  while (i < 255 && s[i])
  {
    if (s[i] >= 'a' && s[i] <= 'z')
      buffer[i] = s[i] - ('a' - 'A');
    else
      buffer[i] = s[i];
    i++;
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

  size_t len_s = __python_strnlen_bounded(s1, 1024);
  size_t len_sub = __python_strnlen_bounded(s2, 1024);

  __ESBMC_assert(len_s <= INT_MAX, "string too long for find()");

  if (len_sub > len_s)
    return -1;

  size_t i = 0;
  while (i + len_sub <= len_s)
  {
    if (strncmp(s1 + i, s2, len_sub) == 0)
      return (int)i;
    i++;
  }

  return -1;
}

int __python_str_find_range(const char *s1, const char *s2, int start, int end)
{
__ESBMC_HIDE:;
  size_t len_s = __python_strnlen_bounded(s1, 1024);
  size_t len_sub = __python_strnlen_bounded(s2, 1024);

  __ESBMC_assert(len_s <= INT_MAX, "string too long for find()");

  __python_str_normalize_range(&start, &end, len_s);

  if (len_sub == 0)
    return (start <= end) ? start : -1;

  if (end - start < (int)len_sub)
    return -1;

  size_t start_u = (size_t)start;
  size_t end_u = (size_t)end;

  size_t i = start_u;
  while (i + len_sub <= end_u)
  {
    if (strncmp(s1 + i, s2, len_sub) == 0)
      return (int)i;
    i++;
  }

  return -1;
}

int __python_str_rfind(const char *s1, const char *s2)
{
__ESBMC_HIDE:;
  size_t len_s = __python_strnlen_bounded(s1, 1024);
  size_t len_sub = __python_strnlen_bounded(s2, 1024);

  __ESBMC_assert(len_s <= INT_MAX, "string too long for rfind()");

  // str.rfind('') = len(s1)
  if (s2[0] == '\0')
    return (int)len_s;

  if (len_sub > len_s)
    return -1;

  size_t start = len_s - len_sub;
  size_t i = start + 1;
  while (i-- > 0)
  {
    if (strncmp(s1 + i, s2, len_sub) == 0)
      return (int)i;
  }

  return -1;
}

int __python_str_rfind_range(const char *s1, const char *s2, int start, int end)
{
__ESBMC_HIDE:;
  size_t len_s = __python_strnlen_bounded(s1, 1024);
  size_t len_sub = __python_strnlen_bounded(s2, 1024);

  __ESBMC_assert(len_s <= INT_MAX, "string too long for rfind()");

  __python_str_normalize_range(&start, &end, len_s);

  if (len_sub == 0)
    return (start <= end) ? end : -1;

  if (end - start < (int)len_sub)
    return -1;

  size_t start_u = (size_t)start;
  size_t end_u = (size_t)end;
  size_t i = end_u - len_sub + 1;

  while (i-- > start_u)
  {
    if (strncmp(s1 + i, s2, len_sub) == 0)
      return (int)i;
  }

  return -1;
}

char *__python_str_replace(
  const char *s,
  const char *old_sub,
  const char *new_sub,
  int count)
{
__ESBMC_HIDE:;
  if (!s || !old_sub || !new_sub)
    return (char *)s;

  if (count == 0)
    return (char *)s;

  // Get string lengths
  size_t old_len = __python_strnlen_bounded(old_sub, 256);
  size_t new_len = __python_strnlen_bounded(new_sub, 256);
  size_t len_s = __python_strnlen_bounded(s, 1024);

  // Bound assumptions for ESBMC - limit string sizes to reasonable values
  __ESBMC_assert(len_s <= 1024, "len_s bounds");
  __ESBMC_assert(old_len <= 256, "old_len bounds");
  __ESBMC_assert(new_len <= 256, "new_len bounds");

  if (old_len == 0)
  {
    size_t slots = len_s + 1;
    size_t replacements = slots;
    if (count > 0 && (size_t)count < slots)
      replacements = (size_t)count;

    size_t result_len = (size_t)len_s + (size_t)replacements * (size_t)new_len;
    char *buffer = __ESBMC_alloca(result_len + 1);

    size_t pos = 0;
    size_t idx = 0;

    while (idx < len_s)
    {
      if (idx < replacements)
      {
        size_t k = 0;
        while (k < new_len)
        {
          buffer[pos] = new_sub[k];
          pos++;
          k++;
        }
      }

      buffer[pos] = s[idx];
      pos++;
      idx++;
    }

    if (len_s < replacements)
    {
      size_t k = 0;
      while (k < new_len)
      {
        buffer[pos] = new_sub[k];
        pos++;
        k++;
      }
    }

    buffer[pos] = '\0';
    return buffer;
  }

  int remaining = count;
  size_t occurrences = 0;
  size_t i = 0;
  while (i + old_len <= len_s)
  {
    if ((remaining != 0) && strncmp(s + i, old_sub, old_len) == 0)
    {
      occurrences++;
      i += old_len;
      if (remaining > 0)
        remaining--;
      if (remaining == 0)
        break;
      continue;
    }
    i++;
  }

  long long diff = (long long)new_len - (long long)old_len;
  long long result_len_signed =
    (long long)len_s + (long long)occurrences * diff;
  if (result_len_signed < 0)
    result_len_signed = 0;
  size_t result_len = (size_t)result_len_signed;
  char *buffer = __ESBMC_alloca(result_len + 1);

  remaining = count;
  i = 0;
  size_t pos = 0;

  // Main replacement loop - use bounded iteration
  while (i < len_s)
  {
    // Check if replacement is possible at current position
    int do_replace = 0;
    if (remaining != 0 && i + old_len <= len_s)
    {
      // Use strncmp for comparison (ESBMC handles this better)
      if (strncmp(s + i, old_sub, old_len) == 0)
        do_replace = 1;
    }

    if (do_replace)
    {
      // Copy new_sub to buffer
      size_t k = 0;
      while (k < new_len)
      {
        buffer[pos] = new_sub[k];
        pos++;
        k++;
      }
      // Skip old_sub in source
      i = i + old_len;
      // Decrement remaining replacements
      if (remaining > 0)
        remaining = remaining - 1;
    }
    else
    {
      // Copy single character
      buffer[pos] = s[i];
      pos++;
      i++;
    }
  }

  buffer[pos] = '\0';
  return buffer;
}

// Python string split - splits a string by separator
// Returns a Python list (represented as PyListObject*)
// For ESBMC, we'll return a simple structure representing the split result
struct __ESBMC_PyListObj *
__python_str_split(const char *str, const char *sep, long long maxsplit)
{
__ESBMC_HIDE:;
  if (!str)
    return (PyListObject *)0;
  if (!sep)
    return (PyListObject *)0;

  size_t len_sep = __python_strnlen_bounded(sep, 64);
  size_t len_str = __python_strnlen_bounded(str, 256);
  PyListObject *out = __ESBMC_list_create();
  if (!out)
    return (PyListObject *)0;

  if (len_sep == 0)
    return out;

  if (maxsplit == 0)
  {
    char *buf = __ESBMC_alloca(len_str + 1);
    memcpy(buf, str, len_str);
    buf[len_str] = '\0';
    __ESBMC_list_push(out, buf, ESBMC_PY_STR_TYPE_ID, len_str + 1);
    return out;
  }

  if (len_sep == 1)
  {
    char sep_ch = sep[0];
    size_t start = 0;
    size_t splits = 0;

    size_t i = 0;
    while (i <= len_str)
    {
      if (i == len_str || str[i] == sep_ch)
      {
        size_t part_len = i - start;
        char *buf = __ESBMC_alloca(part_len + 1);
        size_t k = 0;
        while (k < part_len)
        {
          buf[k] = str[start + k];
          k++;
        }
        buf[part_len] = '\0';
        __ESBMC_list_push(out, buf, ESBMC_PY_STR_TYPE_ID, part_len + 1);

        if (maxsplit > 0)
        {
          splits++;
          if (splits >= (size_t)maxsplit)
          {
            if (i < len_str)
            {
              size_t rem = len_str - (i + 1);
              char *buf2 = __ESBMC_alloca(rem + 1);
              size_t k2 = 0;
              while (k2 < rem)
              {
                buf2[k2] = str[i + 1 + k2];
                k2++;
              }
              buf2[rem] = '\0';
              __ESBMC_list_push(out, buf2, ESBMC_PY_STR_TYPE_ID, rem + 1);
            }
            return out;
          }
        }

        start = i + 1;
      }
      i++;
    }
  }
  else
  {
    char *buf = __ESBMC_alloca(len_str + 1);
    memcpy(buf, str, len_str);
    buf[len_str] = '\0';
    __ESBMC_list_push(out, buf, ESBMC_PY_STR_TYPE_ID, len_str + 1);
  }

  return out;
}

// Python string join - joins a list of strings using a separator
char *__python_str_join(const char *sep, const PyListObject *list)
{
__ESBMC_HIDE:;
  const char *sep_str = sep ? sep : "";
  size_t sep_len = __python_strnlen_bounded(sep_str, 64);

  if (!list)
  {
    char *out = __ESBMC_alloca(1);
    out[0] = '\0';
    return out;
  }

  size_t n = __ESBMC_list_size(list);
  if (n == 0)
  {
    char *out = __ESBMC_alloca(1);
    out[0] = '\0';
    return out;
  }

  // First pass: compute total length
  size_t total = 1; // null terminator
  for (size_t i = 0; i < n; ++i)
  {
    PyObject *item = __ESBMC_list_at((PyListObject *)list, i);
    const char *s = (const char *)item->value;
    if (!s)
      s = "";
    size_t len = 0;
    if (item->size > 0)
    {
      len = item->size;
      if (s[len - 1] == '\0')
        len = len - 1;
      else
        len = __python_strnlen_bounded(s, 1024);
    }
    else
    {
      len = __python_strnlen_bounded(s, 1024);
    }
    total += len;
    if (i + 1 < n)
      total += sep_len;
  }

  char *out = __ESBMC_alloca(total);
  size_t pos = 0;

  for (size_t i = 0; i < n; ++i)
  {
    PyObject *item = __ESBMC_list_at((PyListObject *)list, i);
    const char *s = (const char *)item->value;
    if (!s)
      s = "";
    size_t len = 0;
    if (item->size > 0)
    {
      len = item->size;
      if (s[len - 1] == '\0')
        len = len - 1;
      else
        len = __python_strnlen_bounded(s, 1024);
    }
    else
    {
      len = __python_strnlen_bounded(s, 1024);
    }

    if (len > 0)
    {
      memcpy(out + pos, s, len);
      pos += len;
    }

    if (i + 1 < n && sep_len > 0)
    {
      memcpy(out + pos, sep_str, sep_len);
      pos += sep_len;
    }
  }

  out[pos] = '\0';
  return out;
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

  size_t i = 0;
  size_t pos = 0;

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
