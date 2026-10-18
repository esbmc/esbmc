void *f(void)
{
  void *p;
  p = __builtin_alloca(sizeof(int));
  *(int *)p = 1;
  return p;
}
int main(void)
{
  int *p = f();
  return *p;
}
