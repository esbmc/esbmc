int main()
{
  int a[2][3];
  for (int *p = &a[0][0]; p != &a[2][1]; p++)
    *p = 1;
  return 0;
}
