int main()
{
  unsigned int x = 3, y = 4;

  unsigned int z = 1 + -(x < y); // 1 + -1 = 0
  z / z;                         // division by zero

  return 0;
}