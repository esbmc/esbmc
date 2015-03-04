// count algorithm example
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int main () {
  int mycount;

  // counting elements in array:
  int myints[] = {10,20,30,30,20,10,10,20};   // 8 elements
  mycount = (int) count (myints, myints+8, 10);
  cout << "10 appears " << mycount << " times.\n";

  // counting elements in container:
  vector<int> myvector (myints, myints+8);
  mycount = (int) count (myvector.begin(), myvector.end(), 20);
  cout << "20 appears " << mycount  << " times.\n";

  return 0;
}
