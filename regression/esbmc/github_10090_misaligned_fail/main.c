/* #10090: a store that spans two members at an offset not aligned for its
 * type is still reported. */
#include <assert.h>
struct S
{
  unsigned char c[6];
  unsigned short h;
};
struct S s[2];
int main(void)
{
  unsigned char *p = (unsigned char *)&s[1] + 5;
  *(unsigned short *)p = 0x0102u;
  return 0;
}
