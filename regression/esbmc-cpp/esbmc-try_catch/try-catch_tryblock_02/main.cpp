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
    A() { throw E("Exception in A()"); }
};

class B : public A {
  public:
    B() try : A() { }
    catch(E& e) { cout << "Handler in B(): " << e.error << endl; }
};

int main() {
  try { 
    B cow; 
  }
  catch(E& e) {
    cout << "Handler in main(): " << e.error << endl;
  }
}
 
