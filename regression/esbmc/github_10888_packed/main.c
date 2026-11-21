/* #10888: a packed struct read over a struct of shorts, whose middle member
 * straddles two source members, is rebuilt from the object's bytes. */
#include <assert.h>
struct S
{
  unsigned short x, y;
};
struct __attribute__((packed)) P
{
  unsigned char p;
  unsigned short q;
  unsigned char r;
};
int main(void)
{
  struct S s = {0x0201, 0x0403};
  struct P v = *(struct P *)&s;
  assert(v.p == 1 && v.q == 0x0302 && v.r == 4);
  return 0;
}
