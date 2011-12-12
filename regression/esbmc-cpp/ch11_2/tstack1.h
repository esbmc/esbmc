// Fig. 11.2: tstack1.h
// Stack class template.
#ifndef TSTACK1_H
#define TSTACK1_H

template< class T >
class Stack {

public:
   Stack( int = 10 );  // default constructor (stack size 10)

   // destructor
   ~Stack() 
   { 
      delete [] stackPtr; 
   
   } // end ~Stack destructor

   bool push( const T& );  // push an element onto the stack
   bool pop( T& );         // pop an element off the stack

   // determine whether Stack is empty
   bool isEmpty() const 
   { 
      return top == -1; 
   
   } // end function isEmpty

   // determine whether Stack is full
   bool isFull() const 
   { 
      return top == size - 1; 
   
   } // end function isFull

private:
   int size;     // # of elements in the stack
   int top;      // location of the top element
   T *stackPtr;  // pointer to the stack

}; // end class Stack

// constructor
template< class T >
Stack< T >::Stack( int s )
{
   size = s > 0 ? s : 10;  
   top = -1;  // Stack initially empty
   stackPtr = new T[ size ]; // allocate memory for elements

} // end Stack constructor

// push element onto stack;
// if successful, return true; otherwise, return false
template< class T >
bool Stack< T >::push( const T &pushValue )
{
   if ( !isFull() ) {
      stackPtr[ ++top ] = pushValue;  // place item on Stack
      return true;  // push successful

   } // end if

   return false;  // push unsuccessful

} // end function push

// pop element off stack;
// if successful, return true; otherwise, return false
template< class T > 
bool Stack< T >::pop( T &popValue )
{
   if ( !isEmpty() ) {
      popValue = stackPtr[ top-- ];  // remove item from Stack
      return true;  // pop successful

   } // end if

   return false;  // pop unsuccessful

} // end function pop

#endif

/**************************************************************************
 * (C) Copyright 1992-2003 by Deitel & Associates, Inc. and Prentice      *
 * Hall. All Rights Reserved.                                             *
 *                                                                        *
 * DISCLAIMER: The authors and publisher of this book have used their     *
 * best efforts in preparing the book. These efforts include the          *
 * development, research, and testing of the theories and programs        *
 * to determine their effectiveness. The authors and publisher make       *
 * no warranty of any kind, expressed or implied, with regard to these    *
 * programs or to the documentation contained in these books. The authors *
 * and publisher shall not be liable in any event for incidental or       *
 * consequential damages in connection with, or arising out of, the       *
 * furnishing, performance, or use of these programs.                     *
 *************************************************************************/