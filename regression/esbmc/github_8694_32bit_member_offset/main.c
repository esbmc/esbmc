#include <assert.h>
#include <stddef.h>
struct A { char c; double d; };
struct G { char c; long long ll; };
int main(void)
{
  assert(offsetof(struct A, d) == 4);
  assert(offsetof(struct G, ll) == 4);
  assert(sizeof(struct G) == 12);
  return 0;
}
