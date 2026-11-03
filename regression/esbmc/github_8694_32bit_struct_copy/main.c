#include <assert.h>
#include <string.h>
struct G { char c; long long ll; };
int main(void)
{
  struct G a, b;
  a.c = 1;
  a.ll = 0x0102030405060708LL;
  memcpy(&b, &a, sizeof a);
  assert(b.ll == 0x0102030405060708LL);
  return 0;
}
