// A part the frontend cannot convert fails the whole complex initialiser.
_Atomic int a;

int main(void)
{
  _Complex int z = {1, __c11_atomic_fetch_nand(&a, 1, 0)};
  return 0;
}
