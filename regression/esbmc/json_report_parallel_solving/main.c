#include <assert.h>

int nondet_int(void);

// Twenty independent violations solved on several threads: each one must
// reach report.json exactly once, and the file must stay well-formed.
int main(void)
{
  assert(nondet_int() != 0);
  assert(nondet_int() != 1);
  assert(nondet_int() != 2);
  assert(nondet_int() != 3);
  assert(nondet_int() != 4);
  assert(nondet_int() != 5);
  assert(nondet_int() != 6);
  assert(nondet_int() != 7);
  assert(nondet_int() != 8);
  assert(nondet_int() != 9);
  assert(nondet_int() != 10);
  assert(nondet_int() != 11);
  assert(nondet_int() != 12);
  assert(nondet_int() != 13);
  assert(nondet_int() != 14);
  assert(nondet_int() != 15);
  assert(nondet_int() != 16);
  assert(nondet_int() != 17);
  assert(nondet_int() != 18);
  assert(nondet_int() != 19);
  return 0;
}
