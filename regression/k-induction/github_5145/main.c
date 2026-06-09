// Issue #5145: false alarm with k-induction on aws_hash_table_create_harness.
// __CPROVER_uninterpreted_hasher is defined concretely as (uint64_t)a, which
// maps NULL to 0. The hash table uses 0 as the empty-slot sentinel and
// rejects entries with hash_code == 0.  When key is a nondet void pointer,
// ESBMC explores key = NULL -> hash_code = 0 -> assertion fires, producing a
// false alarm.
//
// CBMC treats __CPROVER_uninterpreted_* functions as truly uninterpreted
// (black-box SMT functions) and does not explore the hash_code == 0 path.
// ESBMC matches this by adding ASSUME(result != 0) after calls to such
// functions, so the spurious path is pruned.
#include <assert.h>
#include <stdint.h>

// SV-COMP harness pattern: concrete body, but CBMC-uninterpreted semantics.
uint64_t __CPROVER_uninterpreted_hasher(const void *a)
{
  return (uint64_t)a;
}

int main(void)
{
  void *key; // nondet — would be NULL without the fix
  uint64_t hash = __CPROVER_uninterpreted_hasher(key);
  // Per CBMC convention, __CPROVER_uninterpreted_* functions are truly
  // uninterpreted: the SMT encoding never maps NULL to 0.  ESBMC adds
  // ASSUME(hash != 0) to match this semantics (issue #5145).
  assert(hash != 0);
  return 0;
}
