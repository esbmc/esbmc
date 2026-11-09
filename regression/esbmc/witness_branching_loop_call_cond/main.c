// A loop whose condition calls a function gets branching waypoints that say
// whether the condition held, not the opposite.
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int a[2];
  int i = 0;
  while (__VERIFIER_nondet_int())
    i++;
  a[i] = 1;
  return 0;
}
