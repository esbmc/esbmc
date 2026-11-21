/* The base case reaches the violation after the loop (esbmc/esbmc#1320). */
int nondet_int();
_Bool nondet_bool();
int x;

int main()
{
  x = nondet_int();
  do
  {
    if (nondet_bool())
      x--;
    else
      x -= 2;
  } while (x > 0);
  __ESBMC_assert(x != 0, "the loop can leave with x == 0");
  return 0;
}
