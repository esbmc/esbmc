#include <ctype.h>
#include <limits.h>
#include <stddef.h>
#include <string.h>
#include "python_types.h"

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

// Python string isalnum - true iff the string is non-empty and every
// character is a letter (ASCII A-Z / a-z) or a digit (0-9). Matches
// Python's CPython behaviour on the ASCII subset; broader Unicode
// category coverage is deliberately out of scope (consistent with the
// existing isalpha/isdigit/isspace models in this file).
_Bool __python_str_isalnum(const char *s)
{
__ESBMC_HIDE:;
  if (!s || !*s)
    return 0;

  while (*s)
  {
    unsigned char c = (unsigned char)*s;
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9')))
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

  // Skip leading chars; return pointer into original string (no copy needed)
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

  return s;
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

// Python string isupper - true iff every cased character is uppercase
// and there is at least one cased character. Mirrors __python_str_islower
// with the cases swapped; UTF-8 continuation bytes after a 0xC2..0xDF
// lead are treated as cased (best-effort Latin-1 supplement coverage).
_Bool __python_str_isupper(const char *s)
{
__ESBMC_HIDE:;
  if (!s || !*s)
    return 0;

  _Bool has_cased = 0;

  while (*s)
  {
    unsigned char c = (unsigned char)*s;

    if (c >= 'a' && c <= 'z')
      return 0;

    if (c >= 'A' && c <= 'Z')
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
  int j = 0;
  while (j < 254 && s[i])
  {
    /* Handle ß (UTF-8: 0xC3 0x9F) -> "SS" */
    if ((unsigned char)s[i] == 0xC3 && (unsigned char)s[i + 1] == 0x9F)
    {
      buffer[j++] = 'S';
      buffer[j++] = 'S';
      i += 2;
    }
    else if (s[i] >= 'a' && s[i] <= 'z')
    {
      buffer[j++] = s[i] - ('a' - 'A');
      i++;
    }
    else
    {
      buffer[j++] = s[i];
      i++;
    }
  }

  buffer[j] = '\0';

  return buffer;
}

// Python string swapcase - returns a new string with ASCII lowercase
// letters uppercased and uppercase letters lowercased; other bytes
// pass through unchanged. Bounded to 255 chars on the receiver, like
// __python_str_lower / _upper, to keep the symbolic loop tractable;
// longer strings trip an explicit assertion rather than silently
// truncating.
char *__python_str_swapcase(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return (char *)s;

  char *buffer = __ESBMC_alloca(256);

  size_t i = 0;
  while (i < 255 && s[i])
  {
    char c = s[i];
    if (c >= 'a' && c <= 'z')
      buffer[i] = c - ('a' - 'A');
    else if (c >= 'A' && c <= 'Z')
      buffer[i] = c + ('a' - 'A');
    else
      buffer[i] = c;
    i++;
  }

  if (s[i] != '\0')
  {
    __ESBMC_assert(
      0, "String too long for swapcase() - exceeds 255 characters");
  }

  buffer[i] = '\0';

  return buffer;
}

// Python string capitalize - returns a copy with the first ASCII letter
// uppercased and every subsequent ASCII letter lowercased. Non-letter
// characters pass through unchanged at their positions. Bounded to 255
// chars on the receiver, like the lower/upper/swapcase models; longer
// strings trip an explicit assertion rather than silently truncating.
char *__python_str_capitalize(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return (char *)s;

  char *buffer = __ESBMC_alloca(256);

  size_t i = 0;
  while (i < 255 && s[i])
  {
    char c = s[i];
    if (i == 0 && c >= 'a' && c <= 'z')
      buffer[i] = c - ('a' - 'A');
    else if (i > 0 && c >= 'A' && c <= 'Z')
      buffer[i] = c + ('a' - 'A');
    else
      buffer[i] = c;
    i++;
  }

  if (s[i] != '\0')
  {
    __ESBMC_assert(
      0, "String too long for capitalize() - exceeds 255 characters");
  }

  buffer[i] = '\0';

  return buffer;
}

// Python string title - returns a copy where the first ASCII letter of
// each word is uppercased and every other letter is lowercased. A word
// starts on any letter that is immediately preceded by a non-letter
// (or by the start of the string); non-letter characters pass through
// unchanged. Bounded to 255 chars on the receiver, same as the
// lower/upper/swapcase/capitalize models.
char *__python_str_title(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return (char *)s;

  char *buffer = __ESBMC_alloca(256);

  _Bool prev_was_letter = 0;
  size_t i = 0;
  while (i < 255 && s[i])
  {
    char c = s[i];
    _Bool is_lower = (c >= 'a' && c <= 'z');
    _Bool is_upper = (c >= 'A' && c <= 'Z');

    if (is_lower || is_upper)
    {
      if (!prev_was_letter)
        buffer[i] = is_upper ? c : (char)(c - ('a' - 'A'));
      else
        buffer[i] = is_lower ? c : (char)(c + ('a' - 'A'));
      prev_was_letter = 1;
    }
    else
    {
      buffer[i] = c;
      prev_was_letter = 0;
    }
    i++;
  }

  if (s[i] != '\0')
  {
    __ESBMC_assert(0, "String too long for title() - exceeds 255 characters");
  }

  buffer[i] = '\0';

  return buffer;
}

// Python string count - count non-overlapping occurrences of `sub` in `s`.
// Matches the Python semantics: empty `sub` returns len(s) + 1 (counts
// gaps including before-first and after-last). Bounded to 256 chars on
// the receiver to keep the symbolic loop tractable for BMC; longer
// strings trip an explicit assertion rather than silently truncating.
size_t __python_str_count(const char *s, const char *sub)
{
__ESBMC_HIDE:;
  if (!s || !sub)
    return 0;

  size_t s_len = __python_strnlen_bounded(s, 256);
  size_t sub_len = __python_strnlen_bounded(sub, 256);

  if (sub_len == 0)
    return s_len + 1;

  if (sub_len > s_len)
    return 0;

  size_t count = 0;
  size_t i = 0;
  while (i + sub_len <= s_len)
  {
    size_t j = 0;
    while (j < sub_len && s[i + j] == sub[j])
      j++;
    if (j == sub_len)
    {
      count++;
      i += sub_len;
    }
    else
    {
      i++;
    }
  }
  return count;
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

// Whitespace predicate matching CPython str.split(None) and the constant-fold
// path in python_list.cpp (space, tab, newline, vertical tab, form feed, CR).
static _Bool __python_str_split_isspace(char c)
{
__ESBMC_HIDE:;
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
         c == '\r';
}

// Copy str[start, end) into a fresh NUL-terminated __ESBMC_alloca buffer and
// append it as the next list element. Each token gets its own allocation (the
// frontend reads element pointers independently); __ESBMC_alloca survives the
// function return (see python_list.cpp), so the buffer stays valid.
static void __python_str_split_emit(
  PyObject *items,
  size_t *count,
  const char *str,
  size_t start,
  size_t end)
{
__ESBMC_HIDE:;
  size_t tok_len = end - start;
  char *buf = __ESBMC_alloca(tok_len + 1);
  size_t k = 0;
  while (k < tok_len)
  {
    buf[k] = str[start + k];
    k++;
  }
  buf[tok_len] = '\0';
  items[*count].value = buf;
  items[*count].type_id = 0;
  items[*count].size = tok_len + 1;
  (*count)++;
}

// Python string split - splits a string by separator and returns a Python list
// (PyListObject*) of the substring tokens. Mirrors CPython semantics and the
// constant-fold path in python_list.cpp::build_split_list:
//   - sep == "" (str.split() / sep=None): split on whitespace runs, drop empty
//     tokens.
//   - non-empty sep: split on each occurrence, keep empty tokens.
//   - maxsplit >= 0 caps the number of splits; the remainder becomes the final
//     token. maxsplit < 0 means unlimited.
//
// The string is traversed with a single monotonic scan (O(n)); each token is
// copied into its own buffer. Keeping the scan linear (rather than re-searching
// from each token start) matters under tight --unwind bounds.
struct __ESBMC_PyListObj *
__python_str_split(const char *str, const char *sep, long long maxsplit)
{
__ESBMC_HIDE:;
  if (!str)
    return (PyListObject *)0;
  if (!sep)
    return (PyListObject *)0;

  size_t len_sep = __python_strnlen_bounded(sep, 64);
  size_t len_str = __python_strnlen_bounded(str, ESBMC_PY_STRNLEN_BOUND);

  // At most len_str + 1 tokens (every character a separator boundary).
  PyObject *items = __ESBMC_alloca((len_str + 1) * sizeof(PyObject));
  PyListObject *list = __ESBMC_alloca(sizeof(PyListObject));
  size_t count = 0;

  if (len_sep == 0)
  {
    // Whitespace split: collapse runs and drop empty tokens.
    long long splits = 0;
    size_t i = 0;
    while (i < len_str)
    {
      while (i < len_str && __python_str_split_isspace(str[i]))
        i++;
      if (i == len_str)
        break;

      size_t start = i;
      if (maxsplit >= 0 && splits >= maxsplit)
      {
        // Remainder is a single token kept verbatim. Leading whitespace was
        // already skipped above; CPython keeps any trailing whitespace here
        // (e.g. 'a  b  '.split(None, 1) == ['a', 'b  ']).
        __python_str_split_emit(items, &count, str, start, len_str);
        break;
      }

      while (i < len_str && !__python_str_split_isspace(str[i]))
        i++;
      __python_str_split_emit(items, &count, str, start, i);
      splits++;
    }
  }
  else
  {
    // Explicit separator: one linear scan, keeping empty tokens.
    long long splits = 0;
    size_t start = 0;
    size_t i = 0;
    while (i + len_sep <= len_str)
    {
      if (maxsplit >= 0 && splits >= maxsplit)
        break;

      size_t j = 0;
      while (j < len_sep && str[i + j] == sep[j])
        j++;

      if (j == len_sep)
      {
        __python_str_split_emit(items, &count, str, start, i);
        i += len_sep;
        start = i;
        splits++;
      }
      else
        i++;
    }

    // Final token: the remainder of the string.
    __python_str_split_emit(items, &count, str, start, len_str);
  }

  list->type = NULL;
  list->items = items;
  list->size = count;
  return list;
}
// Python int() builtin - converts string to integer.
// Returns a 64-bit value: Python ints are modelled as 64-bit everywhere else
// in the frontend (type_handler::get_typet("int")), so a 32-bit return here
// makes the result symbol 32-bit and truncates a string pointer that is later
// rebound through it (e.g. `a, b = s.split('-'); a = int(a)`). See issue #5159.
long long __python_int(const char *s, int base)
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

  long long result = 0;
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

    if (result > (LLONG_MAX / base))
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

// Shared core for float(str): validates `s` as a Python float literal and, when
// valid, writes the parsed value to *out. Returns 1 on success, 0 otherwise.
// The accepted grammar is a subset of CPython's float(): optional surrounding
// ASCII whitespace, an optional sign, and a decimal mantissa with at least one
// digit and at most one '.'. Scientific notation, inf and nan are handled by
// the frontend's compile-time strtod paths (string literals and constant
// symbols); this runtime model deliberately stays a single bounded scan over
// the string so it remains cheap on nondeterministic inputs.
static _Bool __python_parse_float(const char *s, double *out)
{
__ESBMC_HIDE:;
  *out = 0.0;
  if (!s)
    return 0;

  size_t len = __python_strnlen_bounded(s, ESBMC_PY_STRNLEN_BOUND);
  size_t i = 0;

  while (i < len && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' ||
                     s[i] == '\v' || s[i] == '\f' || s[i] == '\r'))
    i++;

  int sign = 1;
  if (i < len && (s[i] == '+' || s[i] == '-'))
  {
    if (s[i] == '-')
      sign = -1;
    i++;
  }

  // Accumulate integer and fractional digits into a single mantissa and divide
  // by 10^(fractional digit count) exactly once at the end. A single rounding
  // step matches std::strtod for short decimal literals, so float("0.3") on a
  // variable agrees with the compile-time strtod path; the alternative of
  // accumulating with a repeatedly-scaled 0.1 weight compounds rounding error.
  double value = 0.0;
  double divisor = 1.0;
  _Bool any_digit = 0;

  while (i < len && s[i] >= '0' && s[i] <= '9')
  {
    value = value * 10.0 + (double)(s[i] - '0');
    any_digit = 1;
    i++;
  }

  if (i < len && s[i] == '.')
  {
    i++;
    while (i < len && s[i] >= '0' && s[i] <= '9')
    {
      value = value * 10.0 + (double)(s[i] - '0');
      divisor *= 10.0;
      any_digit = 1;
      i++;
    }
  }

  if (!any_digit)
    return 0;

  while (i < len && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' ||
                     s[i] == '\v' || s[i] == '\f' || s[i] == '\r'))
    i++;

  if (i != len)
    return 0;

  *out = (double)sign * (value / divisor);
  return 1;
}

// Python float() builtin - converts a string to a double. Returns 0.0 when the
// string is not a valid float literal; callers gate the conversion on
// __python_str_is_float() and raise ValueError on the invalid path.
double __python_str_to_float(const char *s)
{
__ESBMC_HIDE:;
  double value;
  __python_parse_float(s, &value);
  return value;
}

// Returns 1 iff `s` is a valid Python float literal accepted by
// __python_str_to_float, else 0.
_Bool __python_str_is_float(const char *s)
{
__ESBMC_HIDE:;
  double value;
  return __python_parse_float(s, &value);
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

// Python str.join - concatenates the elements of `list` separated by `sep`.
// Empty / NULL list returns "". NULL sep is treated as "". An element whose
// `value` is NULL is silently skipped; CPython raises TypeError there, but the
// rest of the str.* OM family takes the same lenient stance and well-typed
// `List[str]` programs never hit this path. Mirrors the 511-byte fixed-buffer
// pattern used by __python_str_concat: overflow is detected by checking that
// the input was fully drained when an inner loop exits, not by inspecting the
// final pos.
char *__python_str_join(const char *sep, struct __ESBMC_PyListObj *list)
{
__ESBMC_HIDE:;
  char *buffer = __ESBMC_alloca(512);
  buffer[0] = '\0';

  if (!list || list->size == 0)
    return buffer;

  if (!sep)
    sep = "";

  size_t pos = 0;
  size_t i = 0;
  while (i < list->size)
  {
    if (i > 0)
    {
      size_t j = 0;
      while (pos < 511 && sep[j])
      {
        buffer[pos] = sep[j];
        pos++;
        j++;
      }
      __ESBMC_assert(sep[j] == '\0', "join: result exceeds 511 characters");
    }

    const char *s = (const char *)list->items[i].value;
    if (s)
    {
      size_t k = 0;
      while (pos < 511 && s[k])
      {
        buffer[pos] = s[k];
        pos++;
        k++;
      }
      __ESBMC_assert(s[k] == '\0', "join: result exceeds 511 characters");
    }

    i++;
  }

  buffer[pos] = '\0';
  return buffer;
}

// Python string repetition - repeats a string count times
char *__python_str_repeat(const char *s, long long count)
{
__ESBMC_HIDE:;
  if (!s)
    return (char *)0;

  if (count <= 0)
  {
    char *empty = __ESBMC_alloca(1);
    empty[0] = '\0';
    return empty;
  }

  size_t len = __python_strnlen_bounded(s, ESBMC_PY_STRNLEN_BOUND);
  size_t count_u = (size_t)count;

  // Bound checks to keep buffers finite
  __ESBMC_assert(
    count_u <= ESBMC_PY_STRNLEN_BOUND, "String repetition count too large");

  size_t total = len * count_u;
  __ESBMC_assert(
    total <= ESBMC_PY_STRNLEN_BOUND, "String repetition result too large");

  char *buffer = __ESBMC_alloca(total + 1);

  size_t pos = 0;
  size_t i = 0;
  while (i < count_u)
  {
    size_t j = 0;
    while (j < len)
    {
      buffer[pos++] = s[j];
      ++j;
    }
    ++i;
  }

  buffer[pos] = '\0';
  return buffer;
}

char *__python_str_slice(
  const char *s,
  long long start,
  long long end,
  long long step)
{
__ESBMC_HIDE:;
  if (!s)
    return (char *)0;

  size_t len = __python_strnlen_bounded(s, ESBMC_PY_STRNLEN_BOUND);

  // Clamp bounds following Python slice semantics
  if (start < 0)
    start = (long long)len + start;
  if (start < 0)
    start = (step > 0) ? 0 : -1;
  if (start >= 0 && (size_t)start > len)
    start = (step > 0) ? (long long)len : (long long)len - 1;

  if (end < 0)
    end = (long long)len + end;
  if (end < 0)
    end = (step > 0) ? 0 : -1;
  if (end >= 0 && (size_t)end > len)
    end = (long long)len;

  // Calculate result length
  long long result_len = 0;
  if (step > 0 && end > start)
    result_len = (end - start + step - 1) / step;
  else if (step < 0 && start > end)
    result_len = (start - end + (-step) - 1) / (-step);

  if (result_len <= 0)
  {
    char *empty = __ESBMC_alloca(1);
    empty[0] = '\0';
    return empty;
  }

  char *buffer = __ESBMC_alloca((size_t)result_len + 1);

  long long src_idx = start;
  size_t dst_idx = 0;
  while (dst_idx < (size_t)result_len)
  {
    buffer[dst_idx] = s[src_idx];
    src_idx += step;
    dst_idx++;
  }

  buffer[dst_idx] = '\0';
  return buffer;
}

// Python int -> str. Matches str(int) for any 64-bit signed integer:
// decimal digits, leading '-' for negatives, NUL-terminated.
// Buffer fits the widest case: "-9223372036854775808" + NUL = 21 bytes.
char *__python_int_to_str(long long v)
{
__ESBMC_HIDE:;
  char *buffer = __ESBMC_alloca(21);

  if (v == 0)
  {
    buffer[0] = '0';
    buffer[1] = '\0';
    return buffer;
  }

  // Use unsigned magnitude so INT64_MIN is representable without overflow.
  int negative = v < 0;
  unsigned long long mag =
    negative ? -(unsigned long long)v : (unsigned long long)v;

  // Write digits right-to-left into a 20-byte scratch (max 20 decimal digits).
  char digits[20];
  size_t n = 0;
  while (mag > 0)
  {
    digits[n++] = (char)('0' + (mag % 10));
    mag /= 10;
  }

  size_t pos = 0;
  if (negative)
    buffer[pos++] = '-';

  while (n > 0)
    buffer[pos++] = digits[--n];

  buffer[pos] = '\0';
  return buffer;
}

// Python bin(int) -> str. Matches the canonical "0b" prefix; emits "-0b…"
// for negatives. Buffer holds the widest case: '-' + "0b" + 64 binary
// digits + NUL = 68 bytes.
char *__python_int_to_bin(long long v)
{
__ESBMC_HIDE:;
  char *buffer = __ESBMC_alloca(68);

  int negative = v < 0;
  unsigned long long mag =
    negative ? -(unsigned long long)v : (unsigned long long)v;

  // Write binary digits right-to-left.
  char digits[64];
  size_t n = 0;
  if (mag == 0)
    digits[n++] = '0';
  else
    while (mag != 0)
    {
      digits[n++] = (char)('0' + (mag & 1U));
      mag >>= 1U;
    }

  size_t pos = 0;
  if (negative)
    buffer[pos++] = '-';
  buffer[pos++] = '0';
  buffer[pos++] = 'b';

  while (n > 0)
    buffer[pos++] = digits[--n];

  buffer[pos] = '\0';
  return buffer;
}

// Python hex(int) -> str. "0x" prefix, lowercase digits (Python convention).
// Buffer width: '-' + "0x" + 16 hex digits + NUL = 20 bytes.
char *__python_int_to_hex(long long v)
{
__ESBMC_HIDE:;
  char *buffer = __ESBMC_alloca(20);

  int negative = v < 0;
  unsigned long long mag =
    negative ? -(unsigned long long)v : (unsigned long long)v;

  char digits[16];
  size_t n = 0;
  if (mag == 0)
    digits[n++] = '0';
  else
    while (mag != 0)
    {
      unsigned d = (unsigned)(mag & 0xFU);
      digits[n++] = (char)(d < 10 ? '0' + d : 'a' + (d - 10));
      mag >>= 4U;
    }

  size_t pos = 0;
  if (negative)
    buffer[pos++] = '-';
  buffer[pos++] = '0';
  buffer[pos++] = 'x';

  while (n > 0)
    buffer[pos++] = digits[--n];

  buffer[pos] = '\0';
  return buffer;
}

// Python oct(int) -> str. "0o" prefix. Buffer width: '-' + "0o" + 22 octal
// digits + NUL = 26 bytes.
char *__python_int_to_oct(long long v)
{
__ESBMC_HIDE:;
  char *buffer = __ESBMC_alloca(26);

  int negative = v < 0;
  unsigned long long mag =
    negative ? -(unsigned long long)v : (unsigned long long)v;

  char digits[22];
  size_t n = 0;
  if (mag == 0)
    digits[n++] = '0';
  else
    while (mag != 0)
    {
      digits[n++] = (char)('0' + (unsigned)(mag & 7U));
      mag >>= 3U;
    }

  size_t pos = 0;
  if (negative)
    buffer[pos++] = '-';
  buffer[pos++] = '0';
  buffer[pos++] = 'o';

  while (n > 0)
    buffer[pos++] = digits[--n];

  buffer[pos] = '\0';
  return buffer;
}

// Python bool -> str. Returns "True" / "False" with the usual capitalisation.
char *__python_bool_to_str(_Bool b)
{
__ESBMC_HIDE:;
  char *buffer = __ESBMC_alloca(6);
  if (b)
  {
    buffer[0] = 'T';
    buffer[1] = 'r';
    buffer[2] = 'u';
    buffer[3] = 'e';
    buffer[4] = '\0';
  }
  else
  {
    buffer[0] = 'F';
    buffer[1] = 'a';
    buffer[2] = 'l';
    buffer[3] = 's';
    buffer[4] = 'e';
    buffer[5] = '\0';
  }
  return buffer;
}

// Python float -> str. Approximates CPython's str(float) for typical cases:
// integral value -> "X.0", finite non-integral -> shortest "fixed" form with
// trailing zeros stripped, special values -> "nan"/"inf"/"-inf".
// The fixed-precision printout used here matches the existing
// handle_float_to_str() compile-time path (std::to_string + trailing zeros
// stripped).
char *__python_float_to_str(double v)
{
__ESBMC_HIDE:;
  char *buffer = __ESBMC_alloca(64);

  if (v != v)
  {
    buffer[0] = 'n';
    buffer[1] = 'a';
    buffer[2] = 'n';
    buffer[3] = '\0';
    return buffer;
  }

  size_t pos = 0;
  if (v < 0.0)
  {
    buffer[pos++] = '-';
    v = -v;
  }

  // Infinity check: any value larger than the largest representable double
  // after negation is infinite.
  if (v > 1.7976931348623157e+308)
  {
    buffer[pos++] = 'i';
    buffer[pos++] = 'n';
    buffer[pos++] = 'f';
    buffer[pos] = '\0';
    return buffer;
  }

  // Values >= ULLONG_MAX (~1.8e19) cannot be safely cast to unsigned long long
  // (out-of-range float-to-integer conversion is undefined behaviour in C).
  // Emit a fixed-point approximation using pure floating-point arithmetic so
  // the cast below always has a value in [0, ULLONG_MAX).
  // 1.8446744073709551616e19 is the next double above ULLONG_MAX.
  if (v >= 1.8446744073709552e19)
  {
    // Print the integer part digit-by-digit via powers of 10 encoded as double.
    // We emit at most 20 significant digits then append ".0".
    // Use a two-pass approach: determine the order of magnitude, then extract
    // digits top-down using only double arithmetic.
    double scale = 1.0;
    // Find the highest power of 10 <= v (at most 10^308).
    double tmp = v;
    while (tmp >= 10.0)
    {
      tmp /= 10.0;
      scale *= 10.0;
    }
    // tmp is now in [1,10); emit digits
    size_t digit_count = 0;
    while (scale >= 1.0 && digit_count < 20)
    {
      int d = (int)tmp;
      if (d < 0)
        d = 0;
      if (d > 9)
        d = 9;
      buffer[pos++] = (char)('0' + d);
      tmp = (tmp - (double)d) * 10.0;
      scale /= 10.0;
      digit_count++;
    }
    // Always append ".0" to match the "X.0" style for integral values.
    buffer[pos++] = '.';
    buffer[pos++] = '0';
    buffer[pos] = '\0';
    return buffer;
  }

  // Integer part: split off the whole-number portion. Cast is safe because
  // v has been bounded above to be < ULLONG_MAX.
  unsigned long long ip = (unsigned long long)v;
  double frac = v - (double)ip;

  // Write integer digits right-to-left into a 20-byte scratch.
  char digits[20];
  size_t n = 0;
  if (ip == 0)
    digits[n++] = '0';
  while (ip > 0)
  {
    digits[n++] = (char)('0' + (ip % 10));
    ip /= 10;
  }
  while (n > 0)
    buffer[pos++] = digits[--n];

  buffer[pos++] = '.';

  // Fractional part: emit up to 6 digits (matches std::to_string default).
  size_t frac_start = pos;
  size_t i = 0;
  while (i < 6)
  {
    frac *= 10.0;
    int digit = (int)frac;
    if (digit < 0)
      digit = 0;
    if (digit > 9)
      digit = 9;
    buffer[pos++] = (char)('0' + digit);
    frac -= (double)digit;
    i++;
  }

  // Strip trailing zeros from the fractional digits, but keep at least one
  // digit so the result reads like "X.0" rather than "X.".
  while (pos > frac_start + 1 && buffer[pos - 1] == '0')
    pos--;

  buffer[pos] = '\0';
  return buffer;
}
