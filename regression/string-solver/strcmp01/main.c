int main()
{
  int test;
  char strA[7];
  char strB[9];
  strcpy(strA, "01234");
  strcpy(strB, "0123");
  test = strcmp(strA, strB);
  assert(test != 0);
  return 0;
}
