int main()
{
  // A constant non-null integer cast to a pointer is dereferenced without an
  // invalid-pointer check, so this reports SUCCESSFUL. (int *)0 and a nondet
  // integer address are both checked; only the constant escapes.
  int *p = (int *)65;
  return *p;
}
