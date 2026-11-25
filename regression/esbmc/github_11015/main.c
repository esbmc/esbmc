/* #11015: a 4-byte struct that needs only 2-byte alignment, at offset 2. */
#include <assert.h>
struct u2 { unsigned short h, l; };
struct d { unsigned long long x; };
int main(void)
{
  struct d d = {0};
  struct u2 w = {0xffee, 0x1234};
  *(struct u2 *)((unsigned char *)&d + 2) = w;
  struct u2 r = *(struct u2 *)((unsigned char *)&d + 2);
  assert(r.h == 0xffee && r.l == 0x1234);
  return 0;
}
