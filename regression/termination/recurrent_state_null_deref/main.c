// Reading through NULL crashes on the first iteration, so no state recurs.
// With the pointer check dropped, *p would read any value and repeat one.
int main(void)
{
  int *p = 0;
  int i = 0;
  while (i < 3)
    i = *p;
  return 0;
}
