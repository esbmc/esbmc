struct A
{
  int v;
  virtual ~A()
  {
  }
};
struct C : A
{
  ~C()
  {
  }
};

int main()
{
  A *p = new C();
  delete p;
  // Virtual dispatch must not hide the use-after-free.
  return p->v;
}
