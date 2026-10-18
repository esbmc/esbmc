/* #8693: _Complex double is aligned as double, so z sits at offset 8. */
#include <assert.h>
#include <string.h>
struct C
{
  char c;
  _Complex double z;
};
struct D
{
  char c;
  _Complex float z;
  char d;
};
int main(void)
{
  struct C a, b;
  a.c = 1;
  a.z = 2.0;
  memcpy(&b, &a, sizeof a);
  assert(__real__ b.z == 2.0);
  assert((char *)&a.z - (char *)&a == 8);
  struct D d;
  assert((char *)&d.d - (char *)&d == 12);
  return 0;
}
