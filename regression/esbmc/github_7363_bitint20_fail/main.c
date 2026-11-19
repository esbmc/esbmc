int main()
{
  unsigned _BitInt(20) um = 0;
  um = ~um;
  um = um >> 1;
  signed _BitInt(20) m = (signed _BitInt(20))um;
  signed _BitInt(20) one = 1;
  signed _BitInt(20) r = m + one;
  return (int)r;
}
