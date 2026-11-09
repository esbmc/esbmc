#include <assert.h>
#include <stdint.h>
typedef struct { uint32_t cfg; uint32_t shadow[8]; uint32_t tail; } state_t;
static state_t g;
union P { uint32_t *w; char *c; };
int main(void)
{
  unsigned k = nondet_uint();
  if (k >= 8) return 0;
  g.tail = 0;
  union P u;
  u.w = &g.shadow[k];
  char *c = u.c;
  c[(8 - k) * 4] = 42;   /* first byte of g.tail */
  assert(g.tail == 0);
  return 0;
}
