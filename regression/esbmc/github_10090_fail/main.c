/* #10090: the bytes past the array member land in h, low byte first. */
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
  assert(s[i].c[4] == 1 && s[i].c[5] == 2);
  assert(s[i].h == 0x0304);
  return 0;
}
