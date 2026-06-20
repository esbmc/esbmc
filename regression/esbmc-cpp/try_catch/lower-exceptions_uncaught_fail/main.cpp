// An exception that is never caught escapes main: uncaught (std::terminate).
struct X
{
  int v;
  X(int a) : v(a)
  {
  }
};

void f()
{
  throw X(1);
}

int main()
{
  f();
  return 0;
}
