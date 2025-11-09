#include <ctype.h>
#include <limits.h>

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

// Python int() builtin - converts string to integer
// Handles optional base parameter (2-36), with base 10 as default
// Returns 0 for invalid conversions
int __python_int(const char *s, int base)
{
__ESBMC_HIDE:;
  if (!s)
    return 0;

  // Validate base (Python accepts 0, 2-36)
  if (base != 0 && (base < 2 || base > 36))
  {
    __ESBMC_assert(0, "int() base must be >= 2 and <= 36, or 0");
    return 0;
  }

  // Skip leading whitespace
  while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\v' || *s == '\f' ||
         *s == '\r')
    s++;

  // Handle empty string after whitespace
  if (!*s)
  {
    __ESBMC_assert(0, "invalid literal for int() with empty string");
    return 0;
  }

  // Handle sign
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

  // Auto-detect base if base == 0
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
  // Handle explicit base prefixes (0x, 0b, 0o) when base is specified
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

  // Check if we have at least one digit
  if (!*s)
  {
    __ESBMC_assert(0, "invalid literal for int() - no digits");
    return 0;
  }

  // Convert string to integer
  int result = 0;
  _Bool found_digit = 0;

  while (*s)
  {
    int digit_value = -1;
    unsigned char c = (unsigned char)*s;

    // Convert character to digit value
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
      // Trailing whitespace is allowed
      s++;
      continue;
    }
    else
    {
      // Invalid character
      __ESBMC_assert(0, "invalid literal for int() - invalid character");
      return 0;
    }

    // Check if digit is valid for the base
    if (digit_value >= base)
    {
      if (found_digit)
      {
        // We found at least one valid digit, stop here (trailing whitespace OK)
        break;
      }
      __ESBMC_assert(
        0, "invalid literal for int() - digit out of range for base");
      return 0;
    }

    found_digit = 1;

    // Check for overflow (simplified - doesn't handle full range)
    if (result > (INT_MAX / base))
    {
      __ESBMC_assert(0, "int() conversion overflow");
      return 0;
    }

    result = result * base + digit_value;
    s++;
  }

  // Skip trailing whitespace
  while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\v' || *s == '\f' ||
         *s == '\r')
    s++;

  // Check if there are any remaining non-whitespace characters
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
