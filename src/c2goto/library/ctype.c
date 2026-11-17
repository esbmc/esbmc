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

#if defined(__unix__) || defined(__APPLE__) || defined(__MACH__)
#  include <locale.h>
#  if defined(__APPLE__)
#    include <xlocale.h>
#  endif

/* Only the C locale is modelled. */
#  define CTYPE_L(f)                                                           \
    int f##_l(int c, locale_t loc)                                             \
    {                                                                          \
    __ESBMC_HIDE:;                                                             \
      return f(c);                                                             \
    }

#  undef isalnum_l
CTYPE_L(isalnum)
#  undef isalpha_l
CTYPE_L(isalpha)
#  undef isblank_l
CTYPE_L(isblank)
#  undef iscntrl_l
CTYPE_L(iscntrl)
#  undef isdigit_l
CTYPE_L(isdigit)
#  undef isgraph_l
CTYPE_L(isgraph)
#  undef islower_l
CTYPE_L(islower)
#  undef isprint_l
CTYPE_L(isprint)
#  undef ispunct_l
CTYPE_L(ispunct)
#  undef isspace_l
CTYPE_L(isspace)
#  undef isupper_l
CTYPE_L(isupper)
#  undef isxdigit_l
CTYPE_L(isxdigit)
#  undef tolower_l
CTYPE_L(tolower)
#  undef toupper_l
CTYPE_L(toupper)
#endif

/* glibc's ctype.h macros index the class table __ctype_b_loc() points to with
 * any value from -128 to 255. glibc stores each class bit in big-endian order.
 * Only the C locale is modelled, so bytes outside ASCII belong to no class. */
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define CTYPE_BIT(b) (1 << (b))
#else
#  define CTYPE_BIT(b) ((b) < 8 ? (1 << (b)) << 8 : (1 << (b)) >> 8)
#endif
#define CTYPE_UPPER(c) ((c) >= 'A' && (c) <= 'Z')
#define CTYPE_LOWER(c) ((c) >= 'a' && (c) <= 'z')
#define CTYPE_DIGIT(c) ((c) >= '0' && (c) <= '9')
#define CTYPE_ALNUM(c) (CTYPE_UPPER(c) || CTYPE_LOWER(c) || CTYPE_DIGIT(c))
#define CTYPE_GRAPH(c) ((c) > ' ' && (c) < 0x7f)
#define CTYPE_CLASSES(c)                                                       \
  (CTYPE_UPPER(c) * CTYPE_BIT(0) | CTYPE_LOWER(c) * CTYPE_BIT(1) |             \
   (CTYPE_UPPER(c) || CTYPE_LOWER(c)) * CTYPE_BIT(2) |                         \
   CTYPE_DIGIT(c) * CTYPE_BIT(3) |                                             \
   (CTYPE_DIGIT(c) || ((c) >= 'A' && (c) <= 'F') ||                            \
    ((c) >= 'a' && (c) <= 'f')) *                                              \
     CTYPE_BIT(4) |                                                            \
   ((c) == ' ' || ((c) >= '\t' && (c) <= '\r')) * CTYPE_BIT(5) |               \
   ((c) == ' ' || CTYPE_GRAPH(c)) * CTYPE_BIT(6) |                             \
   CTYPE_GRAPH(c) * CTYPE_BIT(7) |                                             \
   ((c) == ' ' || (c) == '\t') * CTYPE_BIT(8) |                                \
   ((c) < ' ' || (c) == 0x7f) * CTYPE_BIT(9) |                                 \
   (CTYPE_GRAPH(c) && !CTYPE_ALNUM(c)) * CTYPE_BIT(10) |                       \
   CTYPE_ALNUM(c) * CTYPE_BIT(11))
#define CTYPE_ROW(c)                                                           \
  CTYPE_CLASSES(c), CTYPE_CLASSES(c + 1), CTYPE_CLASSES(c + 2),                \
    CTYPE_CLASSES(c + 3), CTYPE_CLASSES(c + 4), CTYPE_CLASSES(c + 5),          \
    CTYPE_CLASSES(c + 6), CTYPE_CLASSES(c + 7), CTYPE_CLASSES(c + 8),          \
    CTYPE_CLASSES(c + 9), CTYPE_CLASSES(c + 10), CTYPE_CLASSES(c + 11),        \
    CTYPE_CLASSES(c + 12), CTYPE_CLASSES(c + 13), CTYPE_CLASSES(c + 14),       \
    CTYPE_CLASSES(c + 15)

static const unsigned short ctype_b[384] = {
  [128] = CTYPE_ROW(0),
  CTYPE_ROW(16),
  CTYPE_ROW(32),
  CTYPE_ROW(48),
  CTYPE_ROW(64),
  CTYPE_ROW(80),
  CTYPE_ROW(96),
  CTYPE_ROW(112)};

const unsigned short **__ctype_b_loc(void)
{
__ESBMC_HIDE:;
  static const unsigned short *table = ctype_b + 128;
  return &table;
}
