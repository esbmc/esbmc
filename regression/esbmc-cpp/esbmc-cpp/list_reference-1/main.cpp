#include <iostream>
#include <cassert>
using namespace std;

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

		iterator operator --(){
			this->it = this->it->prev;
			return *this;
		}
		iterator& operator --(int){
			this->it = this->it->prev;
			return *this;
		}

		bool operator ==(const iterator& x) const{
			return (x.it == this->it && x.it_size == this->it_size);
		}
		bool operator !=(const iterator& x) const{
			return (x.it != this->it && x.it_size != this->it_size);
		}

		bool operator <(const iterator&) const;
		bool operator >(const iterator&) const;

		bool operator <=(const iterator&) const;
		bool operator >=(const iterator&) const;

		iterator operator +(int) const;
		iterator operator -(int) const;

		iterator& operator +=(int);
		iterator& operator -=(int);
	};

	explicit list() : head(NULL), tail(NULL), _size(0) {}

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
	iterator insert(iterator position, const T& x){
		node* tmp;
		tmp = new node(x, position.it->prev, position.it);
		position.it->prev->next = tmp;
		position.it->prev = tmp;
		position.it_size++;
		this->_size++;
		return position;
	}
};

int main ()
{
  list<int> mylist;
  list<int>::iterator it;

  mylist.push_back(1);                          // 1 2 3 4 
  mylist.push_back(2);
  mylist.push_back(3);
  mylist.push_back(4);

  it = mylist.begin();
  ++it;       // it points now to number 2           ^

  mylist.insert (it,10);                        // 1 10 2 3 4 

  it = mylist.begin();
  it++;
  it++;
  it++;
  it++;
  assert(*it == 4);
  cout << *it << endl;

  return 0;
}
