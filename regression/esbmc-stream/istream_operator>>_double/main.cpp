// example on extraction
#include <iostream>
#include <cassert>
using namespace std;

int main () {
  double n;
  long double n1;

  cout << "Enter a number: " << endl;
  cin >> n;
  cout << "You have entered: " << n << endl;
  assert((int)cin.gcount() >= 0);
  
  cout << "Enter a number: " << endl;
  cin >> n1;
  cout << "You have entered: " << n1 << endl;
  assert((int)cin.gcount() >= 0);
  
  assert((int)cin.gcount() >= 0);
  return 0;
}
