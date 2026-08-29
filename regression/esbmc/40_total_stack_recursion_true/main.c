int f(int n)
{
  int buf[8];
  buf[0] = n;
  if (n <= 0)
    return buf[0];
  return f(n - 1) + buf[0];
}

int main(void)
{
  return f(3);
}
