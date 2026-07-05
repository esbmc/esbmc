#include <pthread.h>

extern void __VERIFIER_atomic_begin(void);
extern void __VERIFIER_atomic_end(void);

/* Regression for esbmc/esbmc#5133-#5135 (pthread-ext/25_stack and variants).
 *
 * The helper `read_top` reads the shared global `top`, but it is reached only
 * from `__VERIFIER_atomic_check`, a __VERIFIER_atomic_* function whose body
 * goto_convert wraps in an atomic region. Every other access to `top` is also
 * atomic (the writer wraps its write in __VERIFIER_atomic_begin/end). All
 * accesses to `top` therefore hold the global atomic lock, so there is no data
 * race and the expected verdict is VERIFICATION SUCCESSFUL.
 *
 * Before the fix, add_race_assertions instrumented `read_top`'s body in
 * isolation with is_atomic = false, emitting a RACE_CHECK(&top) on the read.
 * That read could observe the writer's in-region atomic-write flag (kept
 * observable for one interleaving point by the #4975 mechanism) and report a
 * spurious R/W data race on `top` -- a false alarm on a correct program.
 *
 * The fix classifies a function whose every call site is atomic (and whose
 * address never escapes) as always-atomic and instruments its body as atomic,
 * exactly as a lexically-wrapped __VERIFIER_atomic_* body.
 */

int top = 0;

int read_top(void) /* reached only from __VERIFIER_atomic_check */
{
  return top; /* shared read -- atomic at every invocation */
}

void __VERIFIER_atomic_check(int *out)
{
  *out = read_top();
}

void *writer(void *arg)
{
  (void)arg;
  __VERIFIER_atomic_begin();
  top = 1; /* atomic write */
  __VERIFIER_atomic_end();
  return 0;
}

void *reader(void *arg)
{
  (void)arg;
  int v;
  __VERIFIER_atomic_check(&v);
  (void)v;
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  reader(0);
  return 0;
}
