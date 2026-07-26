/* Distinct objects passed to restrict-qualified pointers: no aliasing, so
   --restrict-check must not raise an alarm. */
void f(int *restrict a, int *restrict b)
{
  *a = 1;
  *b = 2;
}

int main(void)
{
  int x = 0, y = 0;
  f(&x, &y);
  return 0;
}
