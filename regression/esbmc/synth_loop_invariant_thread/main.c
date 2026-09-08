/* Cutting a loop removes the interleaving points its body carried, so the
 * reader can no longer observe g == 1 or g == 2. The claim is downstream of
 * the havoc, so it is reported UNKNOWN rather than passing silently -- pinned
 * here because the alternative would be a false proof. Under BMC this is
 * VERIFICATION FAILED.
 *
 * __ESBMC_assert, not assert: MSVC spells assert(e) as
 * `(!!(e)) || (_wassert(...), 0)`, whose lowering leaves an `ASSERT 0` guarded
 * by `!e` where glibc and Darwin fold an unguarded `ASSERT e`. #7585's probe
 * asks whether the abstraction still admits the claim holding, and a claim that
 * *is* the constant false answers no on every path, so that spelling would pin
 * the host's <assert.h> rather than the havoc. */
#include <pthread.h>

unsigned int g;

void *writer(void *arg)
{
  unsigned int i = 0;
  g = 0;
  while (i < 3)
  {
    g = g + 1;
    i = i + 1;
  }
  return 0;
}

void *reader(void *arg)
{
  __ESBMC_assert(g == 0 || g == 3, "g == 0 || g == 3");
  return 0;
}

int main(void)
{
  pthread_t t1, t2;
  pthread_create(&t1, 0, writer, 0);
  pthread_create(&t2, 0, reader, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  return 0;
}
