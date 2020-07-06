#include <cstdio>
#include <cassert>

#ifndef DSEXCEPTIONS_H_
#define DSEXCEPTIONS_H_

class Underflow { };
class Overflow  { };
class OutOfMemory { };
class BadIterator
{
public:
  BadIterator()
  {
    goto ERROR;
    ERROR:
    ;
  }

};

#endif

// List class
//
// CONSTRUCTION: with no initializer
// Access is via ListItr class
//
// ******************PUBLIC OPERATIONS*********************
// boolean isEmpty( )     --> Return true if empty; else false
// void makeEmpty( )      --> Remove all items
// ListItr zeroth( )      --> Return position to prior to first
// ListItr first( )       --> Return first position
// void insert( x, p )    --> Insert x after current iterator position p
// void remove( x )       --> Remove x
// ListItr find( x )      --> Return position that views x
// ListItr findPrevious( x )
//                        --> Return position prior to x
// ******************ERRORS********************************
// No special errors



class Memory
{
public:
  int size;
  int *space;
  Memory (int s)
  {
    space = new int[s];
    size = s;
  }

  void destroy ()
  {
    delete[] space;
  }

  ~Memory ()
  {
    if (space)
      delete[] space;
  }
};

class VM
{
public:
  int id;
  Memory *s;
  VM (int i)
  {
    id = i;
    s = NULL;
  }

  void initMemory (Memory * sp)
  {
    this->s = sp;
  }

  void reassignMemory (Memory * sp)
  {
    if (s)
      delete s;
    this->s = sp;
  }

  void destroy ()
  {
    if (s)
      delete s;
  }

  ~VM ()
  {
    destroy();
  }

};

void
printElement (VM * v)
{
  printf (" VM Id:%d VM Memory Size:%d \n", v->id, v->s->size);
}

class List;			// Incomplete declaration.

class ListItr;			// Incomplete declaration.

class ListNode
{
  ListNode ()
  {
    element = NULL;
    next = NULL;
  }
  ListNode (VM * theElement)
  {
    element = theElement;
    next = NULL;
  }
  ListNode (VM * theElement, ListNode * n)
  {
    element = theElement;
    next = n;
  }

  ~ListNode (){
    if(element) delete element;
  }

  VM *element;
  ListNode *next;

  friend class List;
  friend class ListItr;
};


class List
{
public:
  List ();
  List (List & rhs);

  bool isEmpty ();
  void makeEmpty ();
  ListItr zeroth ();
  ListItr first ();
  void insert (VM * &x, ListItr & p);
  ListItr find (VM * &x);
  ListItr findPrevious (VM * &x);
  void remove (VM * &x);

  List & operator= (List & rhs);

private:
  ListNode * header;
};


// ListItr class; maintains "current position"
//
// CONSTRUCTION: Package friendly only, with a ListNode
//
// ******************PUBLIC OPERATIONS*********************
// bool isPastEnd( )      --> True if past end position in list
// void advance( )        --> Advance (if not already null)
// VM* retrieve        --> Return item in current position

class ListItr
{
public:
  ListItr ():current (NULL)
  {
  }
  bool isPastEnd ()
  {
    return current == NULL;
  }
  void advance ()
  {
    if (!isPastEnd ())
      current = current->next;
  }
  VM *&retrieve ()
  {
    if (isPastEnd ())
      throw BadIterator ();
    return current->element;
  }

private:
  ListNode * current;		// Current position

  ListItr (ListNode * theNode):current (theNode)
  {
  }
  friend class List;
};

/**
 * Construct the list
 */
List::List ()
{
  header = new ListNode ();
}

/**
 * Copy ructor
 */
List::List (List & rhs)
{
  header = new ListNode ();
  *this = rhs;
}

/**
 * Test if the list is logically empty.
 * return true if empty, false otherwise.
 */
bool
List::isEmpty ()
{
  return header->next == NULL;
}

/**
 * Make the list logically empty.
 */
void
List::makeEmpty ()
{
  while (!isEmpty ())
    remove (first ().retrieve ());
  delete header;
}

/**
 * Return an iterator representing the header node.
 */
ListItr
List::zeroth ()
{
  return ListItr (header);
}

/**
 * Return an iterator representing the first node in the list.
 * This operation is valid for empty lists.
 */
ListItr
List::first ()
{
  return ListItr (header->next);
}

/**
 * Insert item x after p.
 */
void
List::insert (VM * &x, ListItr & p)
{
  if (p.current != NULL)
  {
    ListNode *tmp = p.current->next;
    p.current->next = new ListNode (x, tmp);
  }
}

/**
 * Return iterator corresponding to the first node containing an item x.
 * Iterator isPastEnd if item is not found.
 */
ListItr
List::find (VM * &x)
{
  /* 1*/ ListNode *itr = header->next;

  /* 2*/ while (itr != NULL && itr->element != x)
    /* 3*/ itr = itr->next;

  /* 4*/ return ListItr (itr);
}

/**
 * Return iterator prior to the first node containing an item x.
 */
ListItr
List::findPrevious (VM * &x)
{
  /* 1*/ ListNode *itr = header;

  /* 2*/ while (itr->next != NULL && itr->next->element != x)
    /* 3*/ itr = itr->next;

  /* 4*/ return ListItr (itr);
}

/**
 * Remove the first occurrence of an item x.
 */
void
List::remove (VM * &x)
{
  ListItr p = findPrevious (x);

  if (p.current->next != NULL)
  {
    ListNode *oldNode = p.current->next;
    p.current->next = p.current->next->next;	// Bypass deleted node
    delete oldNode;
  }
}

/**
 * Deep copy of linked lists.
 */
List & List::operator= (List & rhs)
{
  ListItr
  ritr = rhs.first ();
  ListItr
  itr = zeroth ();

  if (this != &rhs)
  {
    makeEmpty ();
    for (; !ritr.isPastEnd (); ritr.advance (), itr.advance ())
      insert (ritr.retrieve (), itr);
  }

  return *this;
}

#if 0
// Simple print method
template < class VM * >void
printList (List < VM * >&theList)
{
  if (theList.isEmpty ())
    printf ("Empty list\n");
  else
  {
    ListItr < VM * >itr = theList.first ();
    for (; !itr.isPastEnd (); itr.advance ())
      printElement (itr.retrieve ());
  }

}
#endif
void
printList (List & theList)
{
  if (theList.isEmpty ())
    printf ("Empty list\n");
  else
  {
    ListItr
    itr = theList.first ();
    for (; !itr.isPastEnd (); itr.advance ())
      printElement (itr.retrieve ());
  }

}


List
vList;
int
NUM_VMS = 1;

void
createVMs ()
{
  ListItr
  vItr = vList.zeroth ();
  int
  i;
  for (int i = 0; i < NUM_VMS; ++i)
  {
    Memory *
    s = new Memory (1024);
    VM *
    v = new VM (i);
    v->initMemory (s);
    vList.insert (v, vItr);
    printf ("Inserted:");
    printElement (v);
    vItr.advance ();
  }
}

void
resizeVMs ()
{
  ListItr
  vItr = vList.zeroth ();
  for (int i = 0; i <= NUM_VMS; ++i)
  {
    vItr.advance ();
    Memory *
    newM = new Memory (2048);
    VM *
    vm = vItr.retrieve ();
    vm->reassignMemory (newM);
    printf ("Reassigned:");
    printElement (vm);
  }
}

int
main ()
{
  try
  {
    createVMs ();
    //printList(vList);
    resizeVMs ();
    //printList(vList);
  }
  catch (...)
  {
    printf ("Exception caught\n");
    goto ERROR;
    ERROR:
    ;
  }

  vList.makeEmpty();

  return 0;
}
