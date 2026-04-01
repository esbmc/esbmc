/* conc_two_threads_modular_fail:
 * Two threads directly write to the SAME global variable without
 * synchronisation.  This is the reference failing case for data races:
 * no contract abstraction, full body, direct global write.
 *
 * Contrast with conc_two_threads_modular_pass where threads own separate
 * variables and use a precise assigns clause.
 *
 * Expected: VERIFICATION FAILED (W/W data race on counter)
 */
#include <pthread.h>

int counter = 0;

void *t1(void *arg) { counter = 10; return NULL; }
void *t2(void *arg) { counter = 20; return NULL; }

int main()
{
  pthread_t a, b;
  pthread_create(&a, NULL, t1, NULL);
  pthread_create(&b, NULL, t2, NULL);
  pthread_join(a, NULL);
  pthread_join(b, NULL);
  return 0;
}
