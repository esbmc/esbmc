template <class T>
class A
{
  void foo(T t)
  {
  }
};

template <class d>
class g
{
  A<d> f;
};

class i
{
  g<i> h;
};

int main()
{
}
