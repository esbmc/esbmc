#include <string.h>
#include <stdlib.h>

// Python character isalpha - handles ASCII letters only in a single-byte context.
// NOTE: For multi-byte characters (e.g., from iteration), the frontend must
// handle the full multi-byte sequence, as this function is for a single 'int c'
// which is typically only one byte from the string array.
_Bool __python_char_isalpha(int c)
{
__ESBMC_HIDE:;
  // This check is sufficient for *bytes* extracted from a string.
  // The string version handles multi-byte sequences.
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// Python string isalpha - handles ASCII and common two-byte UTF-8 Latin letters.
_Bool __python_str_isalpha(const char *s)
{
__ESBMC_HIDE:;
  if (!s || !*s)
    return 0; // Empty string is not alphabetic

  while (*s)
  {
    unsigned char c = (unsigned char)*s;

    // 1. Handle ASCII letters
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
    {
      s++;
      continue;
    }

    // 2. Reject common non-alphabetic ASCII characters (e.g., space, digit, symbol)
    if (c <= 0x7F)
    {
      // Explicitly reject space (0x20). Python isalpha() must be only letters.
      return 0;
    }

    // 3. Handle Two-Byte UTF-8 sequences (0xC0-0xDF)
    if (c >= 0xC2 && c <= 0xDF)
    {
      unsigned char next = (unsigned char)*(s + 1);
      if (next >= 0x80 && next <= 0xBF)
      {
        // Treat valid two-byte UTF-8 sequences in 0xC2–0xDF as alphabetic.
        // Covers Latin-1 Supplement and common accented letters (é, ñ, ü, etc.).

        s += 2; // Skip both bytes
        continue;
      }
    }

    // 4. Any other sequence (invalid UTF-8, three-byte, four-byte, or unhandled
    // two-byte sequences) is considered non-alphabetic by this model.
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
    return 0; // NULL or empty string returns false

  while (*s)
  {
    if (!isdigit((unsigned char)*s))
      return 0;
    s++;
  }
  return 1; // All characters are digits
}

// Python string isspace: checks if all characters are whitespace
_Bool __python_str_isspace(const char *s)
{
__ESBMC_HIDE:;
  if (!s || !*s)
    return 0; // Empty string is not considered all whitespace in Python

  while (*s)
  {
    unsigned char c = (unsigned char)*s;

    // Check for whitespace: space, tab, newline, vertical tab, form feed, carriage return
    if (!(c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
          c == '\r'))
      return 0;

    s++;
  }

  return 1; // All characters are whitespace
}

// Python string lstrip: removes leading whitespace characters
// Returns a pointer to the first non-whitespace character
const char *__python_str_lstrip(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return s;

  // Skip leading whitespace
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
// Returns true if there's at least one lowercase letter and no uppercase letters
// NOTE: This is a simplified implementation with partial Unicode support!
_Bool __python_str_islower(const char *s)
{
__ESBMC_HIDE:;
  if (!s || !*s)
    return 0; // Empty string returns false

  _Bool has_cased = 0; // Track if we found any cased character

  while (*s)
  {
    unsigned char c = (unsigned char)*s;

    // Check for uppercase ASCII letters
    if (c >= 'A' && c <= 'Z')
      return 0; // Found uppercase, not all lower

    // Check for lowercase ASCII letters
    if (c >= 'a' && c <= 'z')
      has_cased = 1; // Found at least one lowercase letter

    // Handle two-byte UTF-8 sequences for accented letters
    if (c >= 0xC2 && c <= 0xDF)
    {
      unsigned char next = (unsigned char)*(s + 1);
      if (next >= 0x80 && next <= 0xBF)
      {
        // For simplicity, treat valid two-byte UTF-8 as cased characters
        // In real Python, we'd need full Unicode case mapping
        has_cased = 1;
        s += 2;
        continue;
      }
    }

    s++;
  }

  return has_cased; // True only if we found at least one cased character
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
// Uses a static buffer for ESBMC verification
char *__python_str_lower(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return (char *)s;

  // Use a static buffer (sufficient for verification purposes)
  static char buffer[256];

  int i = 0;
  while (i < 255 && s[i])
  {
    if (s[i] >= 'A' && s[i] <= 'Z')
      buffer[i] = s[i] + ('a' - 'A');
    else
      buffer[i] = s[i];
    i++;
  }

  // Warn if string was truncated
  if (s[i] != '\0')
  {
    // String is longer than buffer - issue warning
    __ESBMC_assert(0, "String too long for lower() - exceeds 255 characters");
  }

  buffer[i] = '\0';

  return buffer;
}

// Minimal stub for str.split() - returns a list structure
// Note: This is a placeholder that prevents crashes but doesn't fully implement split
void *__python_str_split(const char *s)
{
__ESBMC_HIDE:;
  // In a real implementation, this would:
  // 1. Count tokens separated by whitespace
  // 2. Allocate a list structure
  // 3. Populate it with the tokens

  // For now, return a non-deterministic pointer
  // This allows verification to continue without full list support
  void *result;
  return result;
}
