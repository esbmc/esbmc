#include <assert.h>
#include <threads.h>

/* Including <threads.h> used to fail outright: ESBMC's -isystem path shadows
 * glibc's <bits/thread-shared-types.h>, so the system <threads.h> could not
 * find the glibc-private types it expects ("unknown type name '__tss_t'"), and
 * platforms shipping no <threads.h> at all reported "file not found".
 * See issue #3449. This exercises the sequential part of the C11 API. */

static int once_calls = 0;
static void init(void)
{
  once_calls++;
}

int main(void)
{
  mtx_t m;
  assert(mtx_init(&m, mtx_plain) == thrd_success);
  assert(mtx_lock(&m) == thrd_success);
  assert(mtx_unlock(&m) == thrd_success);
  mtx_destroy(&m);

  once_flag f = ONCE_FLAG_INIT;
  call_once(&f, init);
  call_once(&f, init);
  assert(once_calls == 1);

  tss_t k;
  int v = 3;
  assert(tss_create(&k, 0) == thrd_success);
  assert(tss_set(k, &v) == thrd_success);
  assert(*(int *)tss_get(k) == 3);

  assert(thrd_equal(thrd_current(), thrd_current()) != 0);

  return 0;
}
