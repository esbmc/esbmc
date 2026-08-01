int main()
{
  // A constant non-null integer cast to a pointer must still be checked; it
  // used to escape the invalid-pointer check (#6544).
  int *p = (int *)65;
  return *p;
}
