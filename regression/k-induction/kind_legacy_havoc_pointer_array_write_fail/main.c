#include <assert.h>
#include <stdint.h>

#define QSIZE 8u

typedef struct
{
  uint32_t wr;
  uint32_t buf[QSIZE];
} queue_t;

unsigned int __VERIFIER_nondet_uint(void);
static queue_t g_q;

static void push(queue_t *const q, uint32_t v)
{
  uint32_t w = q->wr & (QSIZE - 1u);
  q->buf[w] = v;
  q->wr = (q->wr + 1u) & (2u * QSIZE - 1u);
  assert((q->wr & ~(QSIZE - 1u)) == 0u);
}

int main(void)
{
  uint32_t n = __VERIFIER_nondet_uint();
  __ESBMC_assume(n <= 1000000u);
  g_q.wr = 6u;
  for (uint32_t i = 0u; i < n; i++)
    push(&g_q, i);
  return 0;
}
