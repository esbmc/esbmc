int main()
{
  int *p = 0;
  for (int i = 0; i < 2; i++)
  {
    int x = i;
    if (i == 0)
      p = &x;
    else
      return *p;
  }
  return 0;
}
