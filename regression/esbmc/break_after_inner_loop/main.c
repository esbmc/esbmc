int c(void);
int main()
{
  int r = 0;
  while (1)
  {
    int a = 1;
    int *p = &a;
    while (c())
      a++;
    r = *p;
    break;
  }
  return r;
}
