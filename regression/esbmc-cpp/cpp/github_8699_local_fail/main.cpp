// #8699: throwing a local array throws a pointer into the throwing frame
void f()
{
  int a[3] = {1, 2, 3};
  throw a;
}
int main()
{
  try
  {
    f();
  }
  catch (int *p)
  {
    return p[0];
  }
  return 0;
}
