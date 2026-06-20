// Regression test for github.com/esbmc/esbmc/issues/4435.
//
// `construct_from_array`'s dyn-offset alignment check at
// `src/pointer-analysis/dereference.cpp` compared `alignment` (BYTES,
// per `value_set.h:139`) with `subtype_size` (BITS, from
// `type_byte_size_bits`). For an array-of-pointers element reached
// through a struct-member pointer dereference, the unit mismatch
// (alignment = 8 bytes, subtype_size = 64 bits) forced the
// byte-extract / byte-stitch branch instead of the clean `index2tc`
// branch. The byte path then hid the pointer-typed RHS from
// `value_sett::assign`'s array-WITH branch (only the byte_update's
// byte-typed value made it through), so `g_buf[]`'s points-to set
// stayed at `{ NULL }`. On the subsequent read of
// `g_q.buf[0]`, the dereference resolved to `invalid_object` and
// `*p` came back nondet under `--no-pointer-check`.
//
// Pre-fix this single-threaded reproducer FAILS: the final assertion
// (val == 1) tripped because `val` was nondet. The fix unblocks
// pointer-typed array elements specifically, leaving the
// over-conservative byte-extract path for non-pointer subtypes
// (e.g. `regression/esbmc/align-deref_fail`, which still wants the
// alignment check to fire on unaligned int writes).
#include <assert.h>

typedef struct
{
  unsigned int head;
  void **buf;
  unsigned int size;
} queue_t;

static queue_t g_q;
static void *g_buf[4];
static int g_val[4];

int main(void)
{
  g_q.buf = g_buf;
  g_q.size = 4;
  g_val[0] = 1;
  g_q.buf[0 % g_q.size] = &g_val[0];
  void *r = g_q.buf[0];
  assert(r != 0);
  int *p = (int *)r;
  int val = *p;
  assert(val == 1);
  return 0;
}
