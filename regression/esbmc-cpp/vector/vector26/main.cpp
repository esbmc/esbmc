// vector::rbegin/rend
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

// g++ -g -O1 -fsanitize=address -fno-omit-frame-pointer main.cpp -o test_asan
// ./test_asan
// ==22451==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x60300000003c at pc 0x5565ba11ebf9 bp 0x7fff8aec2a20 sp 0x7fff8aec2a10

int main ()
{
  vector<int> myvector;
  for (int i=1; i<=5; i++) myvector.push_back(i);

  cout << "myvector contains:";
  vector<int>::reverse_iterator rit;
  rit = myvector.rend();
  assert(*rit==0); // UNDEFINED BEHAVIOR: Dereferencing rend()
  for ( rit=myvector.rbegin() ; rit < myvector.rend(); ++rit )
    cout << " " << *rit;

  cout << endl;

  return 0;
}
