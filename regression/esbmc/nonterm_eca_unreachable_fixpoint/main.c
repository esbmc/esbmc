// a only ever holds 0 or 1 and both exit; a fixpoint at an unreachable
// value of a (e.g. 5) is no non-terminating execution (#10005).
int a = 0;
extern int __VERIFIER_nondet_int(void);
extern void exit(int);
int update(int input)
{
  if (a == 0 && input == 1) { a = 1; return 1; }
  if (a == 1) exit(0);
  if (a == 0) exit(0);
  return -2;
}
int main(void)
{
  int out = 0;
  while (1)
  {
    int input;
    input = __VERIFIER_nondet_int();
    if (input != 1 && input != 2)
      return -2;
    out = update(input);
  }
  return out;
}
