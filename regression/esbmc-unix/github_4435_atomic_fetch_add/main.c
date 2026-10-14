// Regression test for github.com/esbmc/esbmc/issues/4435.
//
// `__atomic_fetch_add/sub/or/and/xor/nand` (the value variants of the GCC
// `__atomic` family) were silently unimplemented: the outer dispatch in
// `clang_c_adjust_polymorphic_functions.cpp` matched only `__sync_fetch_and_*`,
// so any `__atomic_fetch_*` call fell through and the symex bound its return
// value to a nondet symbol. That made the libvsync SV-COMP benchmark
// `bounded_mpmc_check_full.yml` produce a spurious counterexample where the
// producer wrote to an unknown slot of the bounded queue.
//
// We pin the strong-CAS-style contract of each builtin (`*ptr` updated, OLD
// value returned) so the modelled semantics cannot regress. The AND/OR cases
// also exercise the bitwise vs logical fix: `__sync_fetch_and_and/or` were
// emitting `exprt("and"/"or")` (logical), which trips
// `migrate.cpp` for non-bool operands.
#include <assert.h>

int main(void)
{
  unsigned int v;

  v = 10u;
  unsigned int old = __atomic_fetch_add(&v, 3u, 5);
  assert(old == 10u);
  assert(v == 13u);

  v = 10u;
  old = __atomic_fetch_sub(&v, 4u, 5);
  assert(old == 10u);
  assert(v == 6u);

  v = 0xF0u;
  old = __atomic_fetch_or(&v, 0x0Fu, 5);
  assert(old == 0xF0u);
  assert(v == 0xFFu);

  v = 0xFFu;
  old = __atomic_fetch_and(&v, 0x0Fu, 5);
  assert(old == 0xFFu);
  assert(v == 0x0Fu);

  v = 0xAAu;
  old = __atomic_fetch_xor(&v, 0xFFu, 5);
  assert(old == 0xAAu);
  assert(v == 0x55u);

  return 0;
}
