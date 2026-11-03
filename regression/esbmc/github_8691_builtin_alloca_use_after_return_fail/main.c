int *f(void)
{
  int *p = __builtin_alloca(sizeof(int));
  *p = 1;
  return p;
}
int main(void)
{
  int *p = f();
  return *p;
}
