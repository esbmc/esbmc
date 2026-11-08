/* #10090: on a big-endian target the store's high bytes land in the array. */
#include <assert.h>
struct S
{
  unsigned char c[6];
  unsigned short h;
};
struct S s[2];
int main(void)
{
  unsigned char *p = (unsigned char *)&s[1] + 4;
  *(unsigned int *)p = 0x04030201u;
  assert(s[1].c[4] == 4 && s[1].c[5] == 3 && s[1].h == 0x0201);
  assert(s[1].h == 0x0403);
  return 0;
}
