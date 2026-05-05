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

__ESBMC_contract
int isalnum(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == ((c >= 65 && c <= 90) || (c >= 97 && c <= 122) || (c >= 48 && c <= 57)));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9');
}

// C standard isalpha - ASCII only in default locale
__ESBMC_contract
int isalpha(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == ((c >= 65 && c <= 90) || (c >= 97 && c <= 122)));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

__ESBMC_contract
int isblank(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == (c == 32 || c == 9));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return c == ' ' || c == '\t';
}

__ESBMC_contract
int iscntrl(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == ((c >= 0 && c <= 31) || c == 127));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return (c >= 0 && c <= '\037') || c == '\177';
}

__ESBMC_contract
int isdigit(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == (c >= '0' && c <= '9'));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return c >= '0' && c <= '9';
}

__ESBMC_contract
int isgraph(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == (c >= '!' && c <= '~'));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return c >= '!' && c <= '~';
}

__ESBMC_contract
int islower(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == (c >= 97 && c <= 122));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return c >= 'a' && c <= 'z';
}

__ESBMC_contract
int isprint(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == (c >= ' ' && c <= '~'));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return c >= ' ' && c <= '~';
}

__ESBMC_contract
int ispunct(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~')));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return c == '!' || c == '"' || c == '#' || c == '$' || c == '%' || c == '&' ||
         c == '\'' || c == '(' || c == ')' || c == '*' || c == '+' ||
         c == ',' || c == '-' || c == '.' || c == '/' || c == ':' || c == ';' ||
         c == '<' || c == '=' || c == '>' || c == '?' || c == '@' || c == '[' ||
         c == '\\' || c == ']' || c == '^' || c == '_' || c == '`' ||
         c == '{' || c == '|' || c == '}' || c == '~';
}

__ESBMC_contract
int isspace(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == (c == 32 || c == 9 || c == 10 || c == 11 || c == 12 || c == 13));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r' ||
         c == ' ';
}

__ESBMC_contract
int isupper(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == (c >= 65 && c <= 90));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return c >= 'A' && c <= 'Z';
}

__ESBMC_contract
int isxdigit(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures((__ESBMC_return_value != 0) == ((c >= 48 && c <= 57) || (c >= 65 && c <= 70) || (c >= 97 && c <= 102)));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f') ||
         (c >= '0' && c <= '9');
}

__ESBMC_contract
int tolower(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures(__ESBMC_return_value == ((c >= 65 && c <= 90) ? c + 32 : c));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

__ESBMC_contract
int toupper(int c)
{
    __ESBMC_requires((c >= -1) && (c <= 255));
    __ESBMC_ensures(__ESBMC_return_value == ((c >= 'a' && c <= 'z') ? c - ('a' - 'A') : c));
    __ESBMC_assigns();
__ESBMC_HIDE:;
  return (c >= 'a' && c <= 'z') ? c - ('a' - 'A') : c;
}
