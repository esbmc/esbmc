int more();

int main()
{
  int i = 0;
  while (more())
    ++i;
  for (; more(); ++i)
    ;
  return i;
}
