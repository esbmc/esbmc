// The copied iteration runs at a nondet index, so its branches would pin
// waypoints to the first iteration: the witness keeps only the target.
extern int __VERIFIER_nondet_int(void);
int a[100];
int main()
{
  if (__VERIFIER_nondet_int() > 5)
    return 0;
  for (int i = 0; i < 100; i++)
    if (i % 2)
      a[i] = 100 / (i - 77);
  return 0;
}
