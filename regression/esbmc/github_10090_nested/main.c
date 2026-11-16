/* #10090: an aligned store that starts in a struct member and ends in the
 * next member is not misaligned. */
#include <assert.h>
struct T
{
  unsigned short a, b, c;
};
struct S
{
  struct T t;
  unsigned short d;
};
struct S s[2];
int main(void)
{
  unsigned char *p = (unsigned char *)&s[1] + 4;
  *(unsigned int *)p = 0x00020001u;
  assert(s[1].t.c == 1 && s[1].d == 2);
  assert(s[1].t.b == 0 && s[0].d == 0);
  assert(*(unsigned int *)p == 0x00020001u);
  return 0;
}
