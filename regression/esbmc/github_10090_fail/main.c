/* #10090: the store reaches the member after the one it starts in. */
#include <assert.h>
struct S
{
  unsigned char c[6];
  unsigned short h;
};
struct S s[2];
int main(void)
{
  int i = 1;
  unsigned char *p = (unsigned char *)&s[i] + 4;
  *(unsigned int *)p = 0x04030201u;
  assert(s[i].c[4] == 1 && s[i].c[5] == 2 && s[i].h == 0x0403);
  assert(s[i].h == 0);
  return 0;
}
