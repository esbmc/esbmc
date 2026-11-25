/* As github_10939_nested_src, but asserting the last byte is the first. */
#include <assert.h>
struct d { struct { unsigned char a; } s; unsigned char r[3]; };
union u { unsigned char b[4]; };
int main(void)
{
  struct d d = {{1}, {2, 3, 4}};
  union u v = *(union u *)&d;
  assert(v.b[3] == 1);
  return 0;
}
