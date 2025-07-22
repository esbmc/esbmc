// example on extraction
#include <iostream>
#include <cassert>
using namespace std;

int main () {
  bool n;

  cout << "Enter a bool value: ";
  cin >> boolalpha >> n;
  if (cin.flags() & ios::boolalpha == ios::boolalpha)
    assert(0);
  cout << "You have entered: " << n << endl;

  cout << "Enter another bool value: ";
  cin >> boolalpha >> n;            // manipulator
  if (cin.flags() & ios::boolalpha == ios::boolalpha)
    assert(0);
  cout << "You have entered: " << n << endl;


  return 0;
}
