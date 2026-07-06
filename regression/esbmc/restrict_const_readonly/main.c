/* Both restrict pointers target const objects, so neither can modify the shared
   object; aliasing is well-defined and --restrict-check must not flag it. */
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
