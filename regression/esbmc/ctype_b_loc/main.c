/* What glibc's isspace(c) and friends expand to, as in preprocessed (.i)
 * sv-benchmarks programs, e.g. argv-c/antirez_redis_fastjson. */
#include <assert.h>
#include <ctype.h>

extern const unsigned short int **__ctype_b_loc(void);

enum
{
  is_upper = 1 << 8,
  is_lower = 1 << 9,
  is_alpha = 1 << 10,
  is_digit = 1 << 11,
  is_xdigit = 1 << 12,
  is_space = 1 << 13,
  is_print = 1 << 14,
  is_graph = 1 << 15,
  is_blank = 1 << 0,
  is_cntrl = 1 << 1,
  is_punct = 1 << 2,
  is_alnum = 1 << 3
};

int main()
{
  /* glibc's own C-locale values. */
  const unsigned short *p = *__ctype_b_loc();
  assert(p['A'] == 0xd508 && p[' '] == 0x6001 && p[-1] == 0 && p[255] == 0);
  for (int c = -128; c <= 255; ++c)
  {
    unsigned short b = (*__ctype_b_loc())[c];
    assert(!(b & is_upper) == !isupper(c));
    assert(!(b & is_lower) == !islower(c));
    assert(!(b & is_alpha) == !isalpha(c));
    assert(!(b & is_digit) == !isdigit(c));
    assert(!(b & is_xdigit) == !isxdigit(c));
    assert(!(b & is_space) == !isspace(c));
    assert(!(b & is_print) == !isprint(c));
    assert(!(b & is_graph) == !isgraph(c));
    assert(!(b & is_blank) == !isblank(c));
    assert(!(b & is_cntrl) == !iscntrl(c));
    assert(!(b & is_punct) == !ispunct(c));
    assert(!(b & is_alnum) == !isalnum(c));
  }
  return 0;
}
