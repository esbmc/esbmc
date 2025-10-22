#include <ctype.h>

#undef isalnum
#undef isalpha
#undef isblank
#undef iscntrl
#undef isdigit
#undef isgraph
#undef islower
#undef isprint
#undef ispunct
#undef isspace
#undef isupper
#undef isxdigit
#undef tolower
#undef toupper

int isalnum(int c)
{
__ESBMC_HIDE:;
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9');
}

// C standard isalpha - ASCII only in default locale
int isalpha(int c)
{
__ESBMC_HIDE:;
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

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

int isblank(int c)
{
__ESBMC_HIDE:;
  return c == ' ' || c == '\t';
}

int iscntrl(int c)
{
__ESBMC_HIDE:;
  return (c >= 0 && c <= '\037') || c == '\177';
}

int isdigit(int c)
{
__ESBMC_HIDE:;
  return c >= '0' && c <= '9';
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

int isgraph(int c)
{
__ESBMC_HIDE:;
  return c >= '!' && c <= '~';
}

int islower(int c)
{
__ESBMC_HIDE:;
  return c >= 'a' && c <= 'z';
}

int isprint(int c)
{
__ESBMC_HIDE:;
  return c >= ' ' && c <= '~';
}

int ispunct(int c)
{
__ESBMC_HIDE:;
  return c == '!' || c == '"' || c == '#' || c == '$' || c == '%' || c == '&' ||
         c == '\'' || c == '(' || c == ')' || c == '*' || c == '+' ||
         c == ',' || c == '-' || c == '.' || c == '/' || c == ':' || c == ';' ||
         c == '<' || c == '=' || c == '>' || c == '?' || c == '@' || c == '[' ||
         c == '\\' || c == ']' || c == '^' || c == '_' || c == '`' ||
         c == '{' || c == '|' || c == '}' || c == '~';
}

int isspace(int c)
{
__ESBMC_HIDE:;
  return c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r' ||
         c == ' ';
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

int isupper(int c)
{
__ESBMC_HIDE:;
  return c >= 'A' && c <= 'Z';
}

int isxdigit(int c)
{
__ESBMC_HIDE:;
  return (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f') ||
         (c >= '0' && c <= '9');
}

int tolower(int c)
{
__ESBMC_HIDE:;
  return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

int toupper(int c)
{
__ESBMC_HIDE:;
  return (c >= 'a' && c <= 'z') ? c - ('a' - 'A') : c;
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
