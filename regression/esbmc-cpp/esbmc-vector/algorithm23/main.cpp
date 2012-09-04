// fill algorithm example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

template<class FwdIt, class Ty>
void fill_esbmc(FwdIt first, FwdIt last, const Ty& val) {
	while (first != last)
		*first++ = val;
}

int main () {
  vector<int> myvector (8);                       // myvector: 0 0 0 0 0 0 0 0
  assert(myvector[3] == 0);
  fill_esbmc (myvector.begin(),myvector.begin()+4,5);   // myvector: 5 5 5 5 0 0 0 0
  assert(myvector[3] == 5);
  fill_esbmc (myvector.begin()+3,myvector.end()-2,8);   // myvector: 5 5 5 8 8 8 0 0
  assert(myvector[3] != 8);

  cout << "myvector contains:";
  for (vector<int>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;
 
  return 0;
}
