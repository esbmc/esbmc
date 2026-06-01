extern void fill(int *p);

int main(void)
{
  int a[5];
  fill(&a[0]);
  return a[2];
}
