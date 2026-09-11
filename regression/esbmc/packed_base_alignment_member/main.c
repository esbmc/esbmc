#include <stdint.h>
#include <assert.h>

/* The member access itself is byte-wise and not UB, so it must stay quiet:
   mode.unaligned suppresses the check on this path. */
struct __attribute__((packed)) S
{
  char a;
  char pad[7];
  uint64_t b;
};

int main(void)
{
  struct S sp;
  sp.b = 1;
  uint64_t z = sp.b;
  assert(z == 1);
  return 0;
}
