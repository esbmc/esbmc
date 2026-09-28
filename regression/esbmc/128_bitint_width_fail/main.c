int main()
{
  unsigned _BitInt(80) um = 0;
  um = ~um;
  um = um >> 1;
  /* 2^79 - 1: the largest signed 80-bit value. */
  signed _BitInt(80) m = (signed _BitInt(80))um;
  signed _BitInt(80) one = 1;
  /* Signed overflow at 80 bits. Ranking a 65..127-bit operand INT128 widens
     both sides to 128 bits and the boundary stops being tested. */
  signed _BitInt(80) r = m + one;
  return (int)r;
}
