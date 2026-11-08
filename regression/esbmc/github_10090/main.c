/* #10090: an aligned store that starts near the end of an array member and
 * ends in the next member writes both. */
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
  assert(s[i].c[3] == 0 && s[0].h == 0);
  assert(*(unsigned int *)p == 0x04030201u);
  return 0;
}
