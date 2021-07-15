int main()
{
  unsigned int x, y = 0;
  for(x = 0; x < 100; x++)
  {
    y = y + x;
    if(y > 10000)
      assert(0);
  }
  return 0;
}
