int f(void);
int main(void)
{
  int x = 1;
  int y = x ? f() : 2;
  return y;
}
