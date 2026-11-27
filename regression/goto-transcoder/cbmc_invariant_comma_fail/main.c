/* A comma invariant's value is its right operand, which the loop breaks. */
int a[4];

int main(void)
{
  for (unsigned i = 0; i < 4; i++)
    __CPROVER_loop_invariant((i <= 4, i < 3))
  {
    a[i] = 2;
  }
  return 0;
}
