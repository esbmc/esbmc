// github #6263: deleting through a pointer to a *non-primary* base subobject
// whose class has no virtual destructor is a bad-free ([expr.delete]p3): the
// non-virtual ~Base2 is statically bound, so operator delete receives the
// unadjusted subobject pointer (8 bytes into the Derived allocation) instead
// of the allocation base. ESBMC must report this, not silently accept it.
class Base1
{
public:
  virtual int f(void) { return 21; }
};

class Base2
{
public:
  virtual int g(void) { return 21; }
};

class Derived : public Base1, public Base2
{
public:
  virtual int f(void) { return 42; }
  virtual int g(void) { return 42; }
};

int main()
{
  Base2 *o = new Derived();
  int r = o->g();
  delete o;
  return r;
}
