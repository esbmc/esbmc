// list::sort
//#include <iostream>
//#include <list>
//#include <string>
//#include <cassert>
//using namespace std;
#include <cstddef>
#include <vector>

class A {
  public:
    A();
};

template<class T>
class list {
public:
	struct node {
		T data;
		node* prev;
		node* next;
		node(T t, node* p, node* n) :
				data(t), prev(p), next(n) {
		}
	};

	node* head;
	node* tail;
	int _size;

	class iterator {
	public:
		node* it;
		int it_size;


	iterator end (){
		iterator it;
		//this->tail->next = new node(NULL, this->tail, NULL); //ERRO
		it.it = this->tail->next;
		//it.it_size = this->_size;
		return it;
	}

};
};



int main ()
{
  list<A> mylist;

  return 0;
}
