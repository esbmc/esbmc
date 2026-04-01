/* conc_contract_thread_pass:
 * A thread function processes a shared message buffer.
 * Contract: requires buffer is non-null and size > 0;
 *           ensures the first byte is zeroed (simulates clearing a flag).
 *
 * Verified in isolation (--enforce-contract) — sequential body verification.
 * This is Phase 1 of the modular concurrency workflow:
 *   "Verify each thread function meets its contract before deploying it."
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <stddef.h>

typedef struct {
  unsigned char data[16];
  int size;
  int processed;
} Buffer;

void process_buffer(Buffer *b)
{
  __ESBMC_requires(b != NULL);
  __ESBMC_requires(b->size > 0 && b->size <= 16);
  __ESBMC_ensures(b->data[0] == 0);
  __ESBMC_ensures(b->processed == 1);

  b->data[0] = 0;     /* clear first byte */
  b->processed = 1;   /* mark done */
}

int main() { return 0; }
