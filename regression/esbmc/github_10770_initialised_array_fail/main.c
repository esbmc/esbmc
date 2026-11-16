// The initialiser folds each store into an array constant that also holds the
// earlier calls' values; every call still needs its own waypoint, in order.
extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);

int main(void)
{
  int a[3] = {0};
  for (int i = 0; i < 3; i++)
    a[i] = __VERIFIER_nondet_int();
  if (a[0] == 1 && a[1] == 2 && a[2] == 5)
    reach_error();
  return 0;
}
