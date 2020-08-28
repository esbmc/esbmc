int main()
{
  int i = 2;
  char nomeA[10];
  strcpy(nomeA, "abc");
  strncat(nomeA, "defghi", 2);
  i++;
  assert(nomeA[i] == 'd');
  return 0;
}
