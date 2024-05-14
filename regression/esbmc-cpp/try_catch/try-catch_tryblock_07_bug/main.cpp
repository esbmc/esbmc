#include <iostream>
#include <cassert>
using namespace std;

class E {
public:
  const char* error;
  E(const char* arg) : error(arg) { }
};

class A {
  public: ~A() { throw E("Exception in ~A()"); }
};

class B {
  public: ~B() { throw E("Exception in ~B()"); }
};

int main()
{
try {
  cout << "In main" << endl;
  static A cow;
  B bull;
}
catch (E& e) {
  assert(0);
  cout << e.error << endl;
}
}