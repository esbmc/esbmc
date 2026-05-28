// Regression test for github.com/esbmc/esbmc/issues/4436.
//
// libvsync's HCLH lock (SV-COMP c/libvsync/hclhlock.yml) builds its
// queue-splice on __atomic_compare_exchange_n. Before the fix, ESBMC
// treated that builtin as a no-op: *ptr was never written, *expected
// was never refreshed, and the returned bool was unconstrained. That
// turned every CAS into a silent NOP and made the HCLH acquire path
// see a NULL `pred`, producing a spurious reach_error counterexample
// at k = 5 in the SV-COMP benchmark.
//
// We pin the contract of the strong CAS (both the `_n` value variant
// and the pointer variant) so the modelled semantics cannot regress.
#include <assert.h>
#include <stddef.h>

typedef struct node
{
  int payload;
} node_t;

static node_t a = {1};
static node_t b = {2};
static node_t *head;

int main(void)
{
  // Success path (value variant): *head == NULL == expected, write &a,
  // return true. expected is left unchanged.
  head = NULL;
  node_t *expected = NULL;
  int ok = __atomic_compare_exchange_n(&head, &expected, &a, 0, 5, 5);
  assert(ok);
  assert(head == &a);
  assert(expected == NULL);

  // Failure path (value variant): head now == &a, expected stale NULL.
  // Return false, refresh expected to actual current head value.
  expected = NULL;
  ok = __atomic_compare_exchange_n(&head, &expected, &b, 0, 5, 5);
  assert(!ok);
  assert(head == &a);
  assert(expected == &a);

  // Success path (pointer variant): __atomic_compare_exchange takes
  // `desired` by address. Expected matches → write *desired into *ptr.
  head = NULL;
  expected = NULL;
  node_t *desired = &b;
  ok = __atomic_compare_exchange(&head, &expected, &desired, 0, 5, 5);
  assert(ok);
  assert(head == &b);
  assert(expected == NULL);

  // Failure path (pointer variant): expected stale, refresh from *ptr.
  expected = NULL;
  node_t *other = &a;
  ok = __atomic_compare_exchange(&head, &expected, &other, 0, 5, 5);
  assert(!ok);
  assert(head == &b);
  assert(expected == &b);

  return 0;
}
