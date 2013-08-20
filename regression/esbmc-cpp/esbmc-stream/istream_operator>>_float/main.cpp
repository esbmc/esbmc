// example on extraction
#include <iostream>
#include <cassert>
using namespace std;

int main () {
  float n;

  cout << "Enter a number: ";
  cin >> n;
  cout << "You have entered: " << n << endl;
  
  assert((int)cin.gcount() >= 0);
  return 0;
}
