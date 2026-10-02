// #8006
int main()
{
  int y = 0;
  int *p = &y;
  int a = 0;
  {
    int x = 1;
    p = &x;
    a = *p + 1;
  }
  int b = *p + 1; // x is dead
  return a + b;
}
