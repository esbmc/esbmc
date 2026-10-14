// Exercises all 7 structured-CF code kinds (if/while/for/switch/break/
// continue/label) under --irep2-bodies (V.4.2, esbmc/esbmc#4715).
// Verdict must be VERIFICATION SUCCESSFUL with and without the flag.

int main()
{
  int x = 0;
  int y = 5;

  // code_ifthenelse2t
  if (y > 0)
    x = 1;
  else
    x = 2;

  // code_while2t + code_break2t
  int w = 0;
  while (w < 3)
  {
    if (w == 2)
      break;
    w++;
  }

  // code_for2t + code_continue2t
  int s = 0;
  for (int i = 0; i < 4; i++)
  {
    if (i == 2)
      continue;
    s += i;
  }

  // code_switch2t (+ code_label2t via implicit case labels)
  int z = 0;
  switch (x)
  {
  case 1:
    z = 10;
    break;
  default:
    z = 20;
    break;
  }

  __ESBMC_assert(x == 1, "x should be 1 after if-then-else");
  __ESBMC_assert(w == 2, "while loop with break should stop at w==2");
  __ESBMC_assert(s == 4, "for loop with continue: 0+1+3==4");
  __ESBMC_assert(z == 10, "switch on x==1 should set z=10");

  return 0;
}
