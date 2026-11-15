// The byte 255 of -1 reads back as the signed char -1 under --ir, not 255.
int main(void)
{
  int a = -1;
  __ESBMC_assert(((char *)&a)[0] == 255, "byte as signed char");
  return 0;
}
