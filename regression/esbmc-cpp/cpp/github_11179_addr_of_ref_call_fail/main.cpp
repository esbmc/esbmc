// &at(a) points into a[1], so q[1] is out of bounds.
int &at(int *p)
{
  return p[0];
}

int main()
{
  int a[1] = {0};
  int *q = &at(a);
  return q[1];
}
