// class templates
#include <iostream>
using namespace std;

template <typename T>
class mypair {
    T a, b;
  public:
    mypair (T first, T second)
      {a=first; b=second;}
    virtual T getmax ();
    virtual ~mypair();
};

template <typename T>
T mypair<T>::getmax ()
{
  T retval;
  retval = a>b? a : b;
  return retval;
}

template <typename T>
mypair<T>::~mypair()
{
}


int main () {
  mypair <int> myobject (100, 75);

  mypair <float> myobject2(100.0,75.0);
  cout << myobject.getmax();
  cout << myobject2.getmax();
  return 0;
}
