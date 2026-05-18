#include <iostream>

int compute_index(int x)
{
  volatile int y = x * 17 + 3;
  return (y ^ (y >> 2)) & 7;
}

int main()
{
  int table[8];

  for (int i = 0; i < 8; ++i)
    table[i] = i * 11;

  int total = 0;
  for (int i = 0; i < 8; ++i)
    total += table[compute_index(i)];

  std::cout << total << "\n";
  return 0;
}
