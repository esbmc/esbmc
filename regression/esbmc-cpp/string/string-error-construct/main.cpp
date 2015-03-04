#include <string>
#include <cassert>
#include <iostream>
using namespace std;

int main(){
	string str("esbmc", 10);
	cout << str << endl;
   assert(str.size() == 10);
	return 0;
}
