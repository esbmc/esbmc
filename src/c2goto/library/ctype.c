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
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9');
}

int isalpha(int c)
{
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

int isblank(int c)
{
  return c == ' ' || c == '\t';
}

int iscntrl(int c)
{
  return (c >= 0 && c <= '\037') || c == '\177';
}

int isdigit(int c)
{
  return c >= '0' && c <= '9';
}

int isgraph(int c)
{
  return c >= '!' && c <= '~';
}

int islower(int c)
{
  return c >= 'a' && c <= 'z';
}

int isprint(int c)
{
  return c >= ' ' && c <= '~';
}

int ispunct(int c)
{
  return c == '!' || c == '"' || c == '#' || c == '$' || c == '%' || c == '&' ||
         c == '\'' || c == '(' || c == ')' || c == '*' || c == '+' ||
         c == ',' || c == '-' || c == '.' || c == '/' || c == ':' || c == ';' ||
         c == '<' || c == '=' || c == '>' || c == '?' || c == '@' || c == '[' ||
         c == '\\' || c == ']' || c == '^' || c == '_' || c == '`' ||
         c == '{' || c == '|' || c == '}' || c == '~';
}

int isspace(int c)
{
  return c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r' ||
         c == ' ';
}

int isupper(int c)
{
  return c >= 'A' && c <= 'Z';
}

int isxdigit(int c)
{
  return (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f') ||
         (c >= '0' && c <= '9');
}

int tolower(int c)
{
  return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

int toupper(int c)
{
  return (c >= 'a' && c <= 'z') ? c - ('a' - 'A') : c;
}
