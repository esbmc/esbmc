// Regression test for issue #3928: chained byte-update loop over a struct
// containing a void* field.  After init_loop overwrites all bytes of local_data
// (including the pointer field), __ESBMC_assume constrains the reconstructed
// pointer to equal &some_var.  The assert(0) must be reachable, so the correct
// verdict is VERIFICATION FAILED.
//
// Before the fix (bidirectional int-to-ptr implications), the byte-update chain
// forced intermediate int-to-ptr casts to resolve to the INVALID object because
// the fallback constraint was underconstrained.  This made the assume impossible
// to satisfy and produced a wrong VERIFICATION PASSED result.

char nondet_char();

void init_loop(void *base, unsigned int size)
{
  for (int i = 0; i < size; i++)
    *((char *)base + i) = nondet_char();
}

typedef struct
{
  void *ptr;
} local_t;

int some_var;
local_t local_data;

int main()
{
  init_loop(&local_data, sizeof(local_t));
  __ESBMC_assume(local_data.ptr == &some_var);
  __ESBMC_assert(0, "reachable");
}
