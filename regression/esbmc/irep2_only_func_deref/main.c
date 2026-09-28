/* C11 6.3.2.1p4: dereferencing a pointer to a function yields a function
   designator, which converts straight back to a pointer -- so *f is f, and
   ******f too. Left bare, the code-typed dereference reaches a consumer that
   wants a pointer. */
int f(void)
{
  return 7;
}

int main(void)
{
  int (*p)(void) = &f;

  __ESBMC_assert(*f == &f, "*f is f");
  __ESBMC_assert(***f == &f, "repeated dereference is still f");
  __ESBMC_assert(*p == &f, "through a pointer variable too");
  __ESBMC_assert((*f)() == 7, "and it is still callable");
  return 0;
}
