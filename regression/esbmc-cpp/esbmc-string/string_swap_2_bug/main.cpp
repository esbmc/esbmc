//TEST FAILS
// swap strings
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string buyer ("money");
  string seller ("goods");

  cout << "Before swap, buyer has " << buyer;
  cout << " and seller has " << seller << endl;

  seller.swap (buyer);
  
  assert(seller != "money");
  assert(buyer != "goods");

  cout << " After swap, buyer has " << buyer;
  cout << " and seller has " << seller << endl;

  return 0;
}
