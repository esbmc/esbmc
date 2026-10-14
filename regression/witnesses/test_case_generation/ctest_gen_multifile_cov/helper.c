int clamp(int x)
{
  if (x < 0)
    return 0;
  if (x > 10)
    return 10;
  return x;
}
