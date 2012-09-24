#include <iostream>
using namespace std;
void MyFunc( void );

class CTest {
public:
  CTest() {};
  ~CTest() {};
  const char *ShowReason() const {
    return "Exception in CTest class.";
  }
};

class CDtorDemo {
public:
  CDtorDemo();
  ~CDtorDemo();
};

CDtorDemo::CDtorDemo() {
  cout << "Constructing CDtorDemo.\n";
}

CDtorDemo::~CDtorDemo() {
  cout << "Destructing CDtorDemo.\n";
}

void MyFunc() {
  CDtorDemo D;
  cout<< "In MyFunc(). Throwing CTest exception.\n";
  throw CTest();
}

int main() {
  cout << "In main.\n";
  try {
    cout << "In try block, calling MyFunc().\n";
    MyFunc();
  }
  catch( CTest E ) {
    cout << "In catch handler.\n";
    cout << "Caught CTest exception type: ";
    cout << E.ShowReason() << "\n";
  }
  catch( char *str )    {
    cout << "Caught some other exception: " << str << "\n";
  }
  cout << "Back in main. Execution resumes here.\n";
}
