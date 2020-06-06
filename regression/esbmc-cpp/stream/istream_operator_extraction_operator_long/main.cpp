// example on extraction
#include <iostream>
#include <cassert>
using namespace std;

int main () {
  long n;
  unsigned long n1;
  cout << "Enter a number: ";
  cin >> n;
  cout << "You have entered: " << n << endl;
  assert((int)cin.gcount() >= 0);
  cout << "Enter a hexadecimal number: ";
  cin >> hex >> n;            // manipulator
//  assert(cin.flags() & ios::hex == iostream::hex);
  cout << "Its decimal equivalent is: " << n << endl;
  
  cout << "Enter a number: ";
  cin >> dec >> n1;
  cout << "You have entered: " << n1 << endl;
  assert((int)cin.gcount() >= 0);
  cout << "Enter a hexadecimal number: ";
  cin >> hex >> n1;            // manipulator
//  assert(cin.flags() & ios::hex == iostream::hex);
  cout << "Its decimal equivalent is: " << n1 << endl;
  
  assert((int)cin.gcount() >= 0);
  return 0;
}
