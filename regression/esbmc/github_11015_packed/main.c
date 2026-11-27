/* #11015: packed unions need no alignment at any offset. */
#include <assert.h>
union __attribute__((packed)) U { int i; char c[5]; };
#pragma pack(push, 1)
union V { int i; short s; };
#pragma pack(pop)
struct d { unsigned long long x, y; };
int main(void)
{
  struct d dd = {0, 0};
  union U u;
  u.i = 3;
  *(union U *)((unsigned char *)&dd + 5) = u;
  union V v;
  v.i = 7;
  *(union V *)((unsigned char *)&dd + 3) = v;
  union V r = *(union V *)((unsigned char *)&dd + 3);
  assert(r.i == 7);
  return 0;
}
