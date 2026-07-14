// catch (void*) is the universal pointer catch: it matches a thrown pointer of
// any pointee type.
struct A
{
  int x;
};

int main()
{
  A a;
  a.x = 9;
  try
  {
    throw &a;
  }
  catch (void *p)
  {
    return 1;
  }
  return 0;
}
