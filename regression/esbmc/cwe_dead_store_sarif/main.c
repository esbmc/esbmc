int main(void)
{
  int x = 5; // dead store: overwritten below before any read
  x = 6;
  return x;
}
