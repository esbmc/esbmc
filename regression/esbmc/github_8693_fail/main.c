/* #8693: the copy is complete, but the imaginary part is still zero. */
#include <assert.h>
#include <string.h>
struct C
{
  char c;
  _Complex double z;
};
int main(void)
{
  struct C a, b;
  a.c = 1;
  a.z = 2.0;
  memcpy(&b, &a, sizeof a);
  assert(__real__ b.z == 2.0);
  assert(__imag__ b.z != 0.0);
  return 0;
}
