int a[3];

int main()
{
  int i = 0;
  int s = 0;
  goto test;
body:
  if (a[i] != 0)
    s = s + a[i];
  i = i + 1;
test:
  if (i <= 2)
    goto body;
  return s;
}
