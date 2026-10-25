int c(void);
int main()
{
  int *p = 0;
  while (1)
  {
    int a = 1;
    p = &a;
    while (c())
      a++;
    break;
  }
  return *p;
}
