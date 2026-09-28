// github #6263 (companion): deleting through a pointer to a non-primary base
// subobject IS well-defined when that base has a virtual destructor -- the
// virtual deleting destructor adjusts the interior pointer back to the
// complete object before freeing. ESBMC must keep accepting this; the #6263
// fix narrows the base-subobject relaxation to exactly this case.
struct Base1
{
  int a;
  virtual ~Base1() {}
};

struct Base2
{
  int b;
  virtual ~Base2() {}
};

struct Derived : Base1, Base2
{
  int c;
};

int main()
{
  Base2 *o = new Derived(); // Base2 subobject sits at a non-zero offset
  delete o;                 // virtual ~Base2 -> well-defined
  return 0;
}
