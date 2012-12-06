#include <iostream>
#include <cassert>
using namespace std;

template<class T>
class list {
public:

	typedef bool (pred)(const int &);

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

		iterator(const iterator& x): it(x.it), it_size(x.it_size){}
		iterator(): it(NULL), it_size(0){}
		iterator& operator=(const iterator& x){
			this->it = new node(x.it->data, x.it->prev, x.it->next);
			this->it_size = x.it_size;
			return *this;
		}

		T* operator ->();

		T operator *(){
			return this->it->data;
		}

		iterator operator ++(){
			this->it = this->it->next;
			return *this;
		}
		iterator operator ++(int){
			this->it = this->it->next;
			return *this;
		}
		bool operator ==(const iterator& x) const{
			return (x.it == this->it && x.it_size == this->it_size);
		}
		bool operator !=(const iterator& x) const{
			return (x.it != this->it && x.it_size != this->it_size);
		}
	};

	explicit list (T* t1, T* t2){
		this->_size = 0;
		for(; t1!=t2; t1++)
			this->push_back(*t1);
	}
	iterator begin (){
		iterator it;
		it.it = this->head;
		it.it_size = this->_size;
		return it;
	}
	iterator end (){
		iterator it;
		this->tail->next = new node(NULL, this->tail, NULL);
		it.it = this->tail->next;
		it.it_size = this->_size;
		return it;
	}
	int size() const{
		return this->_size;
	}
	bool empty ( ) const{
		if (this->_size == 0)
			return true;
		return false;
	}
	void push_back ( const T& x ){
		if(this->empty()){
			this->tail = new node(x, NULL, NULL);
			this->head = this->tail;
		}else{
			this->tail->next = new node(x, this->tail, NULL);
			this->tail = this->tail->next;
			if(this->_size == 1)
				this->head->next == this->tail;
		}
		this->_size++;
	}
	void remove_if ( pred* x ){
		int i;
		node* tmp = this->head;
		while(tmp != NULL){
			if (x(tmp->data)){
				this->_size--;
				if(tmp->prev != NULL){
					tmp->prev->next = tmp->next;
				}
				if(tmp->next != NULL){
					tmp->next->prev = tmp->prev;
				}
			}
			tmp = tmp->next;
		}
	}
};

bool single_digit (const int& value) { return (value<10); }

int main ()
{
  int myints[]= {15,36,7,17};
  list<int> mylist (myints,myints+4);   // 15 36 7 17
  list<int>::iterator it;

  mylist.remove_if (single_digit);      // 15 36 17

  assert(mylist.size() == 3);
  it = mylist.begin();
  assert(*it == 15);
  it++;
  assert(*it == 36);
  it++;
  assert(*it == 17);

  return 0;
}
