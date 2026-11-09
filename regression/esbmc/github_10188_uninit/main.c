// #10188: a read through a nondet pointer that is NULL is a NULL dereference.
int main(void)
{
  int *d;
  if (d == 0)
    return *d;
  return 0;
}
