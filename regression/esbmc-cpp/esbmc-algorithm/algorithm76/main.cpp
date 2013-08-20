// copy algorithm example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  int myints[]={10,20,30,40,50};
  vector<int> myvector;	
  vector<int>::iterator it;

  myvector.resize(5);   // allocate space for 7 elements

  copy ( myints, myints+5, myvector.begin() );

  assert(myvector[4] != 50);

  return 0;
}
