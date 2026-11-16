// The first call's branch is fixed by the program text, the second's reads
// x. A validator matches the "true" waypoint to the next execution of line 9,
// the first call, unless that call has a "false" waypoint of its own.
extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);

void __VERIFIER_assert(int cond)
{
  if (!cond)
    reach_error();
}

int main(void)
{
  int x = __VERIFIER_nondet_int();
  __VERIFIER_assert(1);
  __VERIFIER_assert(x != 3);
  return 0;
}
