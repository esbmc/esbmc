// Exercises both dynamic-spec recovery and terminate routing under
// --lower-exceptions. f() throw(X,Y,bad_exception) throws A: unexpected runs
// my_unexpected, which rethrows A (still disallowed) — bad_exception is in the
// spec, so it is substituted and caught by catch(bad_exception&). g() throw(X,Y)
// throws A: unexpected rethrows A, no bad_exception in the spec, so std::terminate
// runs the installed my_terminate, which calls abort() (modelled as silent
// termination). The program therefore verifies SUCCESSFUL.
#include <iostream>
#include<cstdlib>
#include <exception>
using namespace std;

class X { };
class Y { };
class A { };

// pfv type is pointer to function returning void
typedef void (*pfv)();

void my_terminate() {
  cout << "Call to my terminate" << endl;
  abort();
}

void my_unexpected() {
  cout << "Call to my_unexpected()" << endl;
  throw;
}

void f() throw(X,Y, bad_exception) {
  throw A();
}

void g() throw(X,Y) {
  throw A();
}

int main()
{
  pfv old_term = set_terminate(my_terminate);
  pfv old_unex = set_unexpected(my_unexpected);

  try {
    cout << "In first try block" << endl;
    f();
  }
  catch(X) {
    cout << "Caught X" << endl;
  }
  catch(Y) {
    cout << "Caught Y" << endl;
  }
  catch (bad_exception& e1) {
    cout << "Caught bad_exception" << endl;
  }
  catch (...) {
    cout << "Caught some exception" << endl;
  }

  cout << endl;

  try {
    cout << "In second try block" << endl;
    g();
  }
  catch(X) {
    cout << "Caught X" << endl;
  }
  catch(Y) {
    cout << "Caught Y" << endl;
  }
  catch (bad_exception& e2) {
    cout << "Caught bad_exception" << endl;
  }
  catch (...) {
    cout << "Caught some exception" << endl;
  }
}
