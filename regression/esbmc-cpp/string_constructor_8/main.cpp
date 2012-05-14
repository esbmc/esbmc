#include <string>
#include <cassert>

using namespace std;

int main(){
	
	string S = string("Testing");
	string S1 = string("Testing");
	string T = string("Testing", 2);
	assert(T != S + S1);
}
