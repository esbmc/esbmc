int main()
{
  int i = 0;
  int c = 1;
  goto test;
body:
  i = i + 1;
test:
  if (c == 1)
    goto body;
  return 0;
}
