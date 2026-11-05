/* The store through p overwrites the nondet value of t with 0, so the loop
 * always exits: t is not a free choice at the branch. */
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int t;
  int *p = &t;
  while (1)
  {
    t = __VERIFIER_nondet_int();
    *p = 0;
    if (t)
      continue;
    break;
  }
  return 0;
}
