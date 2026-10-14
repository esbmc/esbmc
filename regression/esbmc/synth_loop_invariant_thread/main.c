/* Synthesis declines outright on a program that creates threads: cutting a loop
 * deletes the interleaving points its body carried, so the reader can no longer
 * observe g == 1 or g == 2. Here that would only lose a bug -- the claim stays
 * violable against the havoc, so the refutation path still reports it. The
 * decline is for the shape that does not, which
 * synth_loop_invariant_thread_falseproof pins. This test is the mirror: the
 * verdict a threaded program keeps, and the mechanism line that says why.
 *
 * __ESBMC_assert, not assert: MSVC spells assert(e) as
 * `(!!(e)) || (_wassert(...), 0)`, whose lowering leaves an `ASSERT 0` guarded
 * by `!e` where glibc and Darwin fold an unguarded `ASSERT e`. */
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
