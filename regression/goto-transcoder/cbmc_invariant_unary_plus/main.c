/* Unary plus in a loop invariant is read like any other operator. */
int a[4];

int main(void)
{
  for (unsigned i = 0; i < 4; i++)
    __CPROVER_loop_invariant(+i <= 4)
  {
    a[i] = 0;
  }
  return 0;
}
