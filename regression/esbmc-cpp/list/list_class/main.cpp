#include <iostream>
#include <cassert>
#include <list>
using namespace std;

template<class T>
	class A
{
public:
	A(){
		this->a = true;
	}
	bool get(){
		return this->a;
	}
	void set( bool x ){
		this->a = x;
	}
private:
	bool a;
};

bool operator!=(A<bool> x, A<bool> y){
	if(x.get() != y.get())
		return true;
	return false;
}

bool operator==(A<bool> x, A<bool> y){
	if(x.get() == y.get())
		return true;
	return false;
}

bool operator<(A<bool> x, A<bool> y){
	return false;
}

int main(){
	
	list< A<bool> > x;
	A<bool> a1;
	A<bool> a2;
   a1.set(true);
	a2.set(false);
	x.push_back(a1);
	x.push_back(a2);
	assert((x.front()).get());
	return 0;
}

