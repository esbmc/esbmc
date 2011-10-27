// Definition of Queue class composed of List object
#ifndef QUEUE_C
#define QUEUE_C

#include "list.h"

template< class QUEUETYPE >
class Queue {

public:
   // enqueue calls queueList object's insertAtBack function
   void enqueue( const QUEUETYPE &data ) 
   { 
      queueList.insertAtBack( data ); 
   
   } // end function enqueue

   // dequeue calls queueList object's removeFromFront function
   bool dequeue( QUEUETYPE &data ) 
   { 
      return queueList.removeFromFront( data ); 
   
   } // end function dequeue

   // isQueueEmpty calls queueList object's isEmpty function
   bool isQueueEmpty() const 
   {
      return queueList.isEmpty(); 
   
   } // end function isQueueEmpty

   // printQueue calls queueList object's print function
   void printQueue() const  
   { 
      queueList.print(); 
   
   } // end function printQueue

private:
   List< QUEUETYPE > queueList;  // composed List object

}; // end class Queue

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