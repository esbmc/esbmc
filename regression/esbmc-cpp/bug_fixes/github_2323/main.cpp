namespace
{
  typedef int a;
  template <class>
  struct b;
} // namespace
namespace c
{
  typedef a o;
  template <typename d, typename = b<d>>
  class e;
  template <typename d, typename = b<d>>
  class f;
  typedef f<char> g;
  template <typename d, typename>
  class h
  {
    f<d> *i;
  };
  template <typename d, typename j>
  class f : h<d, j>
  {
    typedef f k;
    template <typename l>
    k &m(l);
  };
  extern template g &g::m(long);
  template <typename d, typename j>
  class e : h<d, j>
  {
    typedef d n;
    typedef e p;
    p &getline(n *, o, n);
  };
  template <>
  e<char> &e<char>::getline(n *, o, n);
} // namespace c

int main()
{
}
