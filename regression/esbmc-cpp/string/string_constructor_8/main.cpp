#include <string>
#include <cassert>
#include <iostream>

using namespace std;

int main(){
	
	string S = string("Te");
	string S1 = string("st");
	string T = string("Testing", 4);
	cout << S << endl << S1 << endl << T << endl;
	assert(T == S + S1);
}
