#include <iostream>
#include <cassert>
using namespace std;

class E {
  public:
    const char* error;
    E(const char* arg) : error(arg) { };
};

class A {
  public:
    A() try { throw E("Exception in A()"); }
    catch(E& e) { cout << "Handler in A(): " << e.error << endl; }
};

int f() try {
  throw E("Exception in f()");
  return 0;
}
catch(E& e) {
  cout << "Handler in f(): " << e.error << endl;
  return 1;
}

int main() {
  int i = 0;
  try { A cow; }
  catch(E& e) {
    cout << "Handler in main(): " << e.error << endl;
  }

  try { i = f(); }
  catch(E& e) {
    cout << "Another handler in main(): " << e.error << endl;
  }

  cout << "Returned value of f(): " << i << endl;
  assert(i==1);
}
