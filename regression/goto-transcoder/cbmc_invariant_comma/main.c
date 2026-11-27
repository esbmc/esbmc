/* goto-cc keeps the comma operator in a loop invariant. */
int a[4];

int main(void)
{
  for (unsigned i = 0; i < 4; i++)
    __CPROVER_loop_invariant((a[0], i <= 4))
  {
    a[i] = 2;
  }
  return 0;
}
