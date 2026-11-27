int main() {
  unsigned char i = 0;
  while (i < 3)
    i++;
  unsigned char j = 0;
  while (j < i)
    j++;
  __ESBMC_assert(j == 2, "after both loops");
  return 0;
}
