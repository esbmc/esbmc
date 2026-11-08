/* #10090: the store reaches the member after the one it starts in. */
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
  *(unsigned int *)p = 0x00070005u;
  assert(s[1].t.c == 5 && s[1].d == 7);
  assert(s[1].d == 0);
  return 0;
}
