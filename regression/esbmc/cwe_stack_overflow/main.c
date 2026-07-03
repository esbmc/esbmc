int main(void)
{
  int a[4];
  a[5] = 42; /* out-of-bounds write on a stack array */
  return a[0];
}
