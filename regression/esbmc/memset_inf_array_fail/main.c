#include <stddef.h>
#include <string.h>
#include <assert.h>

__attribute__((annotate("__ESBMC_inf_size"))) unsigned char base_pool[1];

typedef struct BytesPool
{
  unsigned char *pool;
  size_t pool_cursor;
} BytesPool;

BytesPool pool;

int main()
{
  pool.pool = base_pool;
  pool.pool_cursor = 0;
  memset(&pool.pool[pool.pool_cursor], 1, 7);
  pool.pool_cursor += 7;
  assert(pool.pool[0] == 0);
  return 0;
}
