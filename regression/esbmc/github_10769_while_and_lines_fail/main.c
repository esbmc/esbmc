// A loop condition && with calls is split into one goto per operand.
extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);
int pa(int x) { return x > 0; }
int pb(int x) { return x < 5; }
int main(void)
{
  int x = __VERIFIER_nondet_int();
  int n = 0;
  while (pa(x) &&
         pb(x))
  {
    x = x + 1;
    n = n + 1;
  }
  if (n == 2)
    reach_error();
  return 0;
}
