// Binary node and forward declaration because g++ does
// not understand nested classes.
#include <cstdio>
#include <cassert>
class BinarySearchTree;

class Customer
{
public:
  long ssn;
  Customer ();
  Customer (long s);
  bool operator < (Customer & rhs);
};

Customer::Customer ()
{
  ssn = -1;
}

Customer::Customer (long s)
{
  ssn = s;
}

bool
Customer::operator < (Customer & rhs)
{
  return this->ssn < rhs.ssn;
}

bool
lessthan (Customer * lhs, Customer * rhs)
{
  if (lhs == NULL && rhs != NULL)
    return true;
  else if (lhs != NULL && rhs == NULL)
    return false;
  else if (lhs == rhs && lhs == NULL)
    return false;
  return lhs->ssn < rhs->ssn;
}

void
destroy (Customer * c)
{
  delete c;
}

class BinaryNode
{
  Customer *element;
  BinaryNode *left;
  BinaryNode *right;

  ~BinaryNode ()
  {
    destroy (element);
  }
  BinaryNode (Customer * theElement, BinaryNode * lt,
      BinaryNode * rt):element (theElement), left (lt), right (rt)
  {
  }
  friend class BinarySearchTree;
};


// BinarySearchTree class
//
// CONSTRUCTION: with ITEM_NOT_FOUND object used to signal failed finds
//
// ******************PUBLIC OPERATIONS*********************
// void insert( x )       --> Insert x
// void remove( x )       --> Remove x
// Customer* find( x )   --> Return item that matches x
// Customer* findMin( )  --> Return smallest item
// Customer* findMax( )  --> Return largest item
// boolean isEmpty( )     --> Return true if empty; else false
// void makeEmpty( )      --> Remove all items
// void printTree( )      --> Print tree in sorted order

class BinarySearchTree
{
public:
  explicit BinarySearchTree (Customer * notFound);
  BinarySearchTree (BinarySearchTree & rhs);
  ~BinarySearchTree ();

  Customer *findMin ();
  Customer *findMax ();
  Customer *find (Customer * x);
  bool isEmpty ();
  void printTree ();

  void makeEmpty ();
  void insert (Customer * x);
  void remove (Customer * x);

  BinarySearchTree & operator= (BinarySearchTree & rhs);

private:
  BinaryNode * root;
  Customer *ITEM_NOT_FOUND;

  Customer *elementAt (BinaryNode * t);

  void insertInternal (Customer * x, BinaryNode * &t);
  void removeInternal (Customer * x, BinaryNode * &t);
  BinaryNode *findMin (BinaryNode * t);
  BinaryNode *findMax (BinaryNode * t);
  BinaryNode *find (Customer * x, BinaryNode * t);
  void makeEmptyInternal (BinaryNode * &t);
  void printTreeInternal (BinaryNode * t);
  BinaryNode *clone (BinaryNode * &t);
};

//using namespace std;


class InsertException
{
public:
  Customer * a;
  InsertException ()
  {
    goto ERROR;
    ERROR:
    ;
  }

  InsertException (Customer * x)
  {
    a = x;
    goto ERROR;
    ERROR:
    ;
  }
  ~InsertException ()
  {
  }
};

/**
 * Implements an unbalanced binary search tree.
 * Note that all "matching" is based on the < method.
 */

/**
 * Construct the tree.
 */
BinarySearchTree::BinarySearchTree (Customer * notFound):
    root (NULL), ITEM_NOT_FOUND (notFound)
{
}


/**
 * Copy ructor.
 */
BinarySearchTree::BinarySearchTree (BinarySearchTree & rhs):
    root (NULL), ITEM_NOT_FOUND (rhs.ITEM_NOT_FOUND)
{
  *this = rhs;
}

/**
 * Destructor for the tree.
 */
BinarySearchTree::~BinarySearchTree ()
{
  makeEmpty ();
}

/**
 * Insert x into the tree;
 */
void
BinarySearchTree::insert (Customer * x)
{
  insertInternal (x, root);
}

/**
 * Remove x from the tree. Nothing is done if x is not found.
 */
void
BinarySearchTree::remove (Customer * x)
{
  removeInternal (x, root);
}


/**
 * Find the smallest item in the tree.
 * Return smallest item or ITEM_NOT_FOUND if empty.
 */
Customer *
BinarySearchTree::findMin ()
{
  return elementAt (findMin (root));
}

/**
 * Find the largest item in the tree.
 * Return the largest item of ITEM_NOT_FOUND if empty.
 */
Customer *
BinarySearchTree::findMax ()
{
  return elementAt (findMax (root));
}

/**
 * Find item x in the tree.
 * Return the matching item or ITEM_NOT_FOUND if not found.
 */
Customer *
BinarySearchTree::find (Customer * x)
{
  return elementAt (find (x, root));
}

/**
 * Make the tree logically empty.
 */
void
BinarySearchTree::makeEmpty ()
{
  makeEmptyInternal (root);
}

/**
 * Test if the tree is logically empty.
 * Return true if empty, false otherwise.
 */
bool
BinarySearchTree::isEmpty ()
{
  return root == NULL;
}

/**
 * Print the tree contents in sorted order.
 */
void
BinarySearchTree::printTree ()
{
  if (isEmpty ())
    ;
  else
    printTreeInternal (root);
}

/**
 * Deep copy.
 */
BinarySearchTree & BinarySearchTree::operator= (BinarySearchTree & rhs)
{
  if (this != &rhs)
  {
    makeEmpty ();
    root = clone (rhs.root);
  }
  return *this;
}

/**
 * Internal method to get element field in node t.
 * Return the element field or ITEM_NOT_FOUND if t is NULL.
 */
Customer *
BinarySearchTree::elementAt (BinaryNode * t)
{
  if (t == NULL)
    return ITEM_NOT_FOUND;
  else
    return t->element;
}

/**
 * Internal method to insert into a subtree.
 * x is the item to insert.
 * t is the node that roots the tree.
 * Set the new root.
 */
void
BinarySearchTree::insertInternal (Customer * x, BinaryNode * &t)
{
  if (t == NULL)
    t = new BinaryNode (x, NULL, NULL);
  else
    if (lessthan
        (const_cast < Customer * >(x), const_cast < Customer * >(t->element)))
      insertInternal (x, t->left);
    else
      if (lessthan
          (const_cast < Customer * >(t->element), const_cast < Customer * >(x)))
        insertInternal (x, t->right);
      else {
        throw InsertException (x);	// Duplicate;
      }
}

/**
 * Internal method to remove from a subtree.
 * x is the item to remove.
 * t is the node that roots the tree.
 * Set the new root.
 */
void
BinarySearchTree::removeInternal (Customer * x, BinaryNode * &t)
{
  if (t == NULL)
    return;			// Item not found; do nothing
  if (x < t->element)
    removeInternal (x, t->left);
  else if (t->element < x)
    removeInternal (x, t->right);
  else if (t->left != NULL && t->right != NULL)	// Two children
  {
    t->element = findMin (t->right)->element;
    removeInternal (t->element, t->right);
  }
  else
  {
    BinaryNode *oldNode = t;
    t = (t->left != NULL) ? t->left : t->right;
    delete oldNode;
  }
}

/**
 * Internal method to find the smallest item in a subtree t.
 * Return node containing the smallest item.
 */
BinaryNode *
BinarySearchTree::findMin (BinaryNode * t)
{
  if (t == NULL)
    return NULL;
  if (t->left == NULL)
    return t;
  return findMin (t->left);
}

/**
 * Internal method to find the largest item in a subtree t.
 * Return node containing the largest item.
 */
BinaryNode *
BinarySearchTree::findMax (BinaryNode * t)
{
  if (t != NULL)
    while (t->right != NULL)
      t = t->right;
  return t;
}

/**
 * Internal method to find an item in a subtree.
 * x is item to search for.
 * t is the node that roots the tree.
 * Return node containing the matched item.
 */
BinaryNode *
BinarySearchTree::find (Customer * x, BinaryNode * t)
{
  if (t == NULL)
    return NULL;
  else if (x < t->element)
    return find (x, t->left);
  else if (t->element < x)
    return find (x, t->right);
  else
    return t;			// Match
}

/**
 * Internal method to make subtree empty.
 */
void
BinarySearchTree::makeEmptyInternal (BinaryNode * &t)
{
  if (t != NULL)
  {
    makeEmptyInternal (t->left);
    makeEmptyInternal (t->right);
    //      delete t->element;
    delete t;
  }
  t = NULL;
}

/**
 * Internal method to print a subtree rooted at t in sorted order.
 */
void
BinarySearchTree::printTreeInternal (BinaryNode * t)
{
  if (t != NULL)
  {
    printTreeInternal (t->left);
    printTreeInternal (t->right);
  }
}

/**
 * Internal method to clone subtree.
 */
BinaryNode *
BinarySearchTree::clone (BinaryNode * &t)
{
  if (t == NULL)
    return NULL;
  else
    return new BinaryNode (t->element, clone (t->left), clone (t->right));
}


int
main ()
{
  BinarySearchTree t (NULL);
  int NUMS = 400;
  int GAP = 37;
  int i;

  try
  {
    long ssn1 = 10243;
    Customer *c = new Customer (long (ssn1));
    t.insert (c);
    printf ("Inserted %d\n", ssn1);

    long ssn2 = 10244;
    c = new Customer (long (ssn2));
    t.insert (c);
    printf ("Inserted %d\n", ssn2);

    long ssn3 = 10244;
    c = new Customer (long (ssn3));
    t.insert (c);
    printf ("Inserted %d\n", ssn3);

  }
  catch (InsertException x)
  {
    printf ("Could not create Customer:%d\n", (x.a)->ssn);
  }


  t.makeEmpty ();
  return 0;
}
