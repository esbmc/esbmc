/*
 * Mode C — C-Live proof for the `if (!ended)` branch in `__pyt_join`
 *
 * Production source: src/c2goto/library/pthread_lib.c:1014-1021
 *
 * The branch body (lines 1016-1020) increments blocked_threads_count and
 * asserts the non-deadlock invariant.  This harness proves the branch is
 * reachable by constructing the canonical "join before terminate" state:
 *   - tid 1 is initialised (running, not ended)
 *   - num_threads_running == 2  (main thread + spawned thread)
 *   - blocked_threads_count == 0
 * and then calling the inlined __pyt_join logic.
 *
 * Contract preconditions (G3):
 *   - __ESBMC_assume(tid == 1)
 *       pthread state arrays are indexed 0..num_total_threads-1;
 *       tid 1 is the first spawned thread (src/c2goto/library/pthread_lib.c:227-231).
 *   - __ESBMC_assume(!ended)
 *       caller-side precondition: join called before terminate fires;
 *       this is the exact condition the branch guards against
 *       (pthread_lib.c:1013-1014).
 *   - __ESBMC_assume(num_threads_running == 2)
 *       one main thread (tid 0) + one spawned thread (tid 1) running
 *       (pthread_lib.c:227-228: pthread_create bumps num_threads_running).
 *   - __ESBMC_assume(blocked_threads_count == 0)
 *       no threads blocked before this join (invariant on entry).
 *
 * Stubs (G4): none needed — harness is self-contained.
 *
 * G5 (sanity variant): replace the __ESBMC_unreachable() call with
 *   __ESBMC_assert(0, "sanity") and confirm VERIFICATION FAILED.
 */

/* Reproduce the globals visible to __pyt_join inline.
 * The OM uses these names via the flail.py-mangled symbol table, but for
 * a standalone harness we declare them locally and use them directly. */
_Bool ended_arr[2];                    /* pthread_thread_ended[tid]       */
unsigned short int num_running;        /* num_threads_running             */
unsigned short int blocked;            /* blocked_threads_count           */

/* The __pyt_join logic, inlined from pthread_lib.c:1009-1025,
 * with __ESBMC_unreachable() inserted inside the `if (!ended)` body
 * to prove C-Live. */
static void pyt_join_inlined(unsigned int thread)
{
  _Bool ended = ended_arr[thread];
  if (!ended)
  {
    /* C-Live instrumentation: if we reach here, the branch is reachable. */
    __ESBMC_unreachable();

    blocked++;
    /* Deadlock check (production assertion) */
    __ESBMC_assert(
      blocked != num_running,
      "Deadlocked state in __pyt_join");
  }
  /* Block until ended (ASSUME simulates the symex blocking primitive) */
  __ESBMC_assume(ended);
}

int main(void)
{
  unsigned int tid = 1u;

  /* G3: tid in range — first spawned thread */
  __ESBMC_assume(tid < 2u);

  /* G3: join-before-terminate scenario — ended flag not yet set */
  ended_arr[tid] = 0; /* thread tid has NOT yet called __pyt_terminate */

  /* G3: two threads running (main + spawned), none blocked yet */
  num_running = 2;
  blocked     = 0;

  pyt_join_inlined(tid);

  return 0;
}
