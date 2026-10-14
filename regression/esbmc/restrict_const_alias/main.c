/* const targets are not exempt: C11 6.7.3.1p4 makes aliasing undefined once the
   shared object is modified by any means, which cannot be ruled out here, so
   two aliasing const restrict pointers are still flagged. */
int g;

void f(const int *restrict a, const int *restrict b)
{
  g = *a + *b;
}

int main(void)
{
  int x = 1;
  f(&x, &x);
  return 0;
}
