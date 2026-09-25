int sink;

void g(void)
{
  int a[8];
  a[0] = 1;
  sink += a[0];
}

int main(void)
{
  g();
  g();
  g();
  g();
  return 0;
}
