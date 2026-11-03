int main()
{
  unsigned _BitInt(40) um = 0;
  um = ~um;
  um = um >> 1;
  signed _BitInt(40) m = (signed _BitInt(40))um;
  signed _BitInt(40) one = 1;
  signed _BitInt(40) r = m + one;
  return (int)r;
}
