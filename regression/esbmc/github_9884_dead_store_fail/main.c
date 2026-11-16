// #9884: without propagation the branch reads a[0] directly. The slicer keeps
// that array version but drops the store a[0] = x as dead, so the branch must
// get no waypoint.
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int a[2];
  int y = 0;
  int x = __VERIFIER_nondet_int();
  a[0] = x;
  a[1] = x;
  if (a[0] > 5)
    y = 1;
  return a[1] + 1;
}
