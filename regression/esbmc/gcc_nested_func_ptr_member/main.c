// Nested function reads/writes a captured struct pointer via `->`.
// Exercises the tokenizer's single-token handling of the `->` operator.
struct S
{
  int x;
  int y;
};

int main()
{
  struct S s = {1, 2};
  struct S *p = &s;

  void bump()
  {
    p->x += 10;
    p->y = p->x + p->y;
  }

  bump();
  __ESBMC_assert(p->x == 11, "bump via -> updated x");
  __ESBMC_assert(p->y == 13, "bump via -> updated y");
  return 0;
}
