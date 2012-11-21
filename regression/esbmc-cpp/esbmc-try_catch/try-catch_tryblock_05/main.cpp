#include <iostream>
using namespace std;

class E {
  public:
    const char* error;
    E(const char* arg) : error(arg) { };
};

class B {
  public:
    B() { };
    ~B() { cout << "~B() called" << endl; };
};

class D : public B {
  public:
    D();
    ~D() { cout << "~D() called" << endl; };
};

D::D() try : B() {
  throw E("Exception in D()");
}
catch(E& e) {
  cout << "Handler of function try block of D(): " << e.error << endl;
};

int main() {
  try {
    D val;
  }
  catch(...) { }
}
