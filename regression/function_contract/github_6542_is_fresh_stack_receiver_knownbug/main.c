/* github_6542_is_fresh_stack_receiver_knownbug:
 * A caller may legitimately pass a live automatic object (#6380), and the
 * extent conjunct is guarded by is_dynamic so it does not reject one:
 * __ESBMC_alloc_size is maintained for the heap only, and the guard is visible
 * in the emitted predicate.
 *
 * This still fails, on valid_object alone. __ESBMC_alloc is never updated for
 * stack pointers (see the note in goto-symex/dynamic_allocation.cpp), so
 * VALID_OBJECT of an automatic object is a free boolean a solver may pick
 * false. That is #6542's automatic-storage half, which is open and separate
 * from the extent this test guards. */
typedef struct { int coeffs[4]; } P;

void callee(P *p) {
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(P)));
  __ESBMC_assigns(p->coeffs);
  __ESBMC_ensures(p->coeffs[0] == 1);
  p->coeffs[0] = 1;
}

int main(void) {
  P v;
  callee(&v);
  return 0;
}
