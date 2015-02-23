// example on extraction
#include <iostream>
#include <cassert>
using namespace std;

int main () {
  bool n;

  cout << "Enter a bool value: ";
  cin >> boolalpha >> n;
  assert(cin.flags() & ios::boolalpha == ios::boolalpha);
  cout << "You have entered: " << n << endl;

  cout << "Enter another bool value: ";
  cin >> boolalpha >> n;            // manipulator
  assert(cin.flags() & ios::boolalpha == ios::boolalpha);
  cout << "You have entered: " << n << endl;


  return 0;
}
