// Fig. 17.12: stackcomposition.h
// Template Stack class definition with composed List object.
#ifndef STACKCOMPOSITION
#define STACKCOMPOSITION

#include "list.h"  // List class definition

template< class STACKTYPE >
class Stack {

public:
   // no constructor; List constructor does initialization

   // push calls stackList object's insertAtFront function
   void push( const STACKTYPE &data ) 
   { 
      stackList.insertAtFront( data ); 
   
   } // end function push
   
   // pop calls stackList object's removeFromFront function
   bool pop( STACKTYPE &data ) 
   { 
      return stackList.removeFromFront( data ); 
   
   } // end function pop
   
   // isStackEmpty calls stackList object's isEmpty function
   bool isStackEmpty() const 
   { 
      return stackList.isEmpty(); 
   
   } // end function isStackEmpty
   
   // printStack calls stackList object's print function
   void printStack() const 
   { 
      stackList.print(); 
   
   } // end function printStack

private:
   List< STACKTYPE > stackList;  // composed List object

}; // end class Stack

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