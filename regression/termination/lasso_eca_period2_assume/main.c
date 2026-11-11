// Real: a == 1 is cut by the assume; input 2 from a == 0 exits. No infinite
// execution exists.
int a = 0;
extern int __VERIFIER_nondet_int(void);
extern void exit(int);
extern void __VERIFIER_assume(int);
int update(int input)
{
  if (a == 0 && input == 1) { a = 1; return 1; }
  if (a == 1 && input == 2) { a = 0; return 2; }
  exit(0);
  return -2;
}
int main(void)
{
  int sink = 0;
  while (1)
  {
    __VERIFIER_assume(a != 1);
    int input;
    input = __VERIFIER_nondet_int();
    if (input != 1 && input != 2)
      return -2;
    sink = update(input);
  }
  return sink;
}
