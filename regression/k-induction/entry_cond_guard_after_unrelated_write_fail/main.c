// The inner loop's exit test reads j after j++. The step may assume it only
// for a state before the loop that passes it: when m <= 3 the inner loop
// leaves in its first iteration, and n still reaches 20.
int nondet_int();
int main()
{
  int m = nondet_int();
  int n = 0;
  while (1)
  {
    n++;
    int j = 0;
    while (1)
    {
      j++;
      if (!(m > 3 && j < 10))
        break;
    }
    __ESBMC_assert(!(m <= 3 && n == 20), "n != 20");
  }
}
