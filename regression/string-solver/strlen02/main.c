int main()
{
  int i;
  char strA[7];
  strcpy(strA, "012");
  i = strlen(strA);
  assert(i == 3);
  assert(strA[1] == '2');
  return 0;
}
