// `int (*r)[N]` does name a whole array, so the lift fires, and it takes the
// same route a directly named global array takes -- both reach
// lift_old_over_bound_index with an array-typed object and build
// `dereference(typecast(__ESBMC_old_raw(&object), int (*)[N]))`. The named
// global verifies; this one reports
//   ERROR: Can't construct rvalue reference to array type during dereference
//
// So reading the snapshot back as an array rvalue is not what fails: the global
// does that too. The two differ only in what `&object` is. For the global it is
// the address of a named object; here `object` is `*r`, so it is `&*r`, an
// address computed through a pointer, and the dereference layer cannot
// construct the array rvalue for it. Confirmed by instrumenting the lift:
//   named global   object.id=symbol       object.type=array   SUCCESSFUL
//   pointer to arr object.id=dereference  object.type=array   the error above
//
// Two ways round it were tried and both trade the error for a crash, so the
// shape is pinned rather than half-fixed:
//   - snapshot through the pointer and index the element type, `((int *)snap)[j]`:
//     assertion `!is_scalar_type(expr)' in
//     dereferencet::dereference_expr_nonscalar (dereference.cpp:529)
//   - the same with explicit pointer arithmetic, `*((int *)snap + j)`:
//     assertion in assert_arith_2ops_consistency (irep2_expr.cpp:673), the
//     index needing a width the construction does not give it
//
// What this needs is for the snapshot to be taken of a pointed-to region rather
// than of a named object; see github_4219_old_in_forall_array_param_knownbug,
// which needs the same thing for a different reason.
#define N 4
#define BOUND 100

void bump(int (*r)[N])
{
  unsigned j;
  __ESBMC_requires(__ESBMC_is_fresh(r, sizeof(int[N])));
  __ESBMC_requires(
    __ESBMC_forall(&j, !(j < N) || ((*r)[j] > -BOUND && (*r)[j] < BOUND)));
  __ESBMC_ensures(
    __ESBMC_forall(&j, !(j < N) || ((*r)[j] == __ESBMC_old((*r)[j]) + 1)));
  __ESBMC_assigns(*r);

  for (unsigned i = 0; i < N; i++)
    (*r)[i] = (*r)[i] + 1;
}

int main(void)
{
  return 0;
}
