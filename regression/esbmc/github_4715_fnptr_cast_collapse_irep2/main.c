int g(int x)
{
  return x + 1;
}

int main(void)
{
  /* The cast is a no-op once the function designator decays. The default path
     emits `p=&g`; the IREP2 adjust pass keeps the cast
     (scope-clang-c-irep2.md 143). */
  int (*p)(int) = (int (*)(int))g;
  return p(1) == 2 ? 0 : 1;
}
