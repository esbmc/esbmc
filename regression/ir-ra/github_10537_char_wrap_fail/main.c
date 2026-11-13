/* The unwrapped value is still checked: 255 + 1 is 256, not 0. */
int main()
{
  unsigned char c = 255;
  c = c + 1;
  __ESBMC_assert((int)c == 0, "wrapped value");
  return 0;
}
