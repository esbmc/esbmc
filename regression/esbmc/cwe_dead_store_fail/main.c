int main(void)
{
  int x = 5; // dead store: overwritten below before any read
  x = 6;
  return 1 / (x - 6); // division by zero: x - 6 == 0
}
