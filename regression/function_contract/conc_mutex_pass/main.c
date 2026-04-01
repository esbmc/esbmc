/* conc_mutex_pass:
 * Two threads increment a shared counter, each protected by a mutex.
 * Contract on inc_counter expresses the atomic-increment postcondition.
 * Step 1 (this test): enforce the contract on inc_counter in isolation —
 *   verifies the body satisfies the postcondition.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <pthread.h>
#include <assert.h>
#include <stddef.h>

int counter = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void inc_counter(int *c)
{
  __ESBMC_requires(c != NULL);
  __ESBMC_ensures(*c == __ESBMC_old(*c) + 1);
  pthread_mutex_lock(&lock);
  (*c)++;
  pthread_mutex_unlock(&lock);
}

int main() { return 0; }
