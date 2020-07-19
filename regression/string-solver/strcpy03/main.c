int main()
{
  char strA[7];
  strcpy(strA, "0123");
  assert(strA[3] == '3');
  strcpy(strA, "abcde");
  assert(strA[4] == 'e');
  return 0;
}
