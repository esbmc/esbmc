int main()
{
  const int ARRAY_SIZE = 10;
  long packet[ARRAY_SIZE];
  packet[20] = 1; // out-of-bounds write
  return 0;
}
