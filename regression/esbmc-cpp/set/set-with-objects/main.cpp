#include <iostream>
#include <cassert>
#include <set>
#include <cstdlib>
using namespace std;

class Ball {
public:
	int radius;
	int weight;
	int circunference;
	
	Ball(){
	  radius = 0;
	  weight = 0;
	  circunference = 0;
	}
	
	Ball(int r, int w){
	  radius = r;
	  weight = w;
	  circunference = 2*3.14*r;
	}
	
	bool operator<(Ball b){
		return (circunference < b.circunference);
	}
	bool operator==(Ball b){
		return (circunference == b.circunference);
	}
	Ball operator=(Ball b){
	  radius = b.radius;
	  weight = b.weight;
	  circunference = b.circunference;
	  return *this;
	}
};

int main(){
  set<Ball> myset;
  set<Ball>::iterator it;

  Ball soccer(10,10);
  Ball basket(15,20);
  myset.insert(soccer);
  myset.insert(basket);
  
  myset.insert(Ball(1,10));
  it = myset.find(soccer);
  assert(*it == soccer); 
  cout << endl;
  return 0;
}
