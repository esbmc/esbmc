extern void reach_error(void);
int main()
{
  unsigned i = 0;
  while (1)
  {
    i++;
    if (i == 5)
      reach_error();
  }
}
