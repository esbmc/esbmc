// iter_swap example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

template<class Ty>
void swap_esbmc(Ty& a, Ty& b) {
	Ty c(a);
	a = b;
	b = c;
}

template<class ForwardIterator1, class ForwardIterator2>
void iter_swap_esbmc(ForwardIterator1 a[], ForwardIterator2 b) {
	swap_esbmc(*a, *b);
}

template<class ForwardIterator1, class ForwardIterator2>
void iter_swap_esbmc(ForwardIterator1 a, ForwardIterator2 b) {
	swap_esbmc(*a, *b);
}

int main () {

  int myints[]={10,20,30,40,50 };          //   myints:  10  20  30  40  50
  vector<int> myvector (4,99);             // myvector:  99  99  99  99

  iter_swap_esbmc(myints,myvector.begin());      //   myints: [99] 20  30  40  50
                                           // myvector: [10] 99  99  99

  iter_swap_esbmc(myints+3,myvector.begin()+2);  //   myints:  99  20  30 [99]
                                           // myvector:  10  99 [40] 99

  assert(myvector[0] == 10);
  assert(myvector[1] == 99);
  assert(myvector[2] == 40);
  assert(myvector[3] == 99);

  cout << "myvector contains:";
  for (vector<int>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
