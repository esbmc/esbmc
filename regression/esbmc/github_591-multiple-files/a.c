int *p = &(int){0};
extern int *q;
int main()
{
  *q = 5;
  assert(*p == 0);
}
