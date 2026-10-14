/* Aliasing restrict pointers, but --restrict-check is NOT passed: the pass is
   opt-in, so this must verify successfully (no restrict assertion added). */
void f(int *restrict a, int *restrict b)
{
  *a = 1;
  *b = 2;
}

int main(void)
{
  int x = 0;
  f(&x, &x);
  return 0;
}
