// An eca-rers2012-shaped loop: with a == 1 and input == 2 no branch of
// calculate_output fires, so that state repeats forever.
extern int __VERIFIER_nondet_int(void);
int a = 1;
int calculate_output(int input)
{
  if (a == 1 && input == 1)
  {
    a = 2;
    return 0;
  }
  if (a == 2 && input == 2)
  {
    a = 1;
    return 0;
  }
  return -2;
}
int main(void)
{
  int output = -1;
  while (1)
  {
    int input;
    input = __VERIFIER_nondet_int();
    if ((input != 1) && (input != 2))
      return -2;
    output = calculate_output(input);
  }
}
