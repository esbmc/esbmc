// Fig. 17.13: queue.h
// Template Queue class definition derived from class List.
#ifndef QUEUE_H
#define QUEUE_H

#include "list.h"  // List class definition

template< class QUEUETYPE >
class Queue: private List< QUEUETYPE > {

public:
   // enqueue calls List function insertAtBack
   void enqueue( const QUEUETYPE &data ) 
   { 
      insertAtBack( data ); 
   
   } // end function enqueue

   // dequeue calls List function removeFromFront
   bool dequeue( QUEUETYPE &data ) 
   { 
      return removeFromFront( data ); 
   
   } // end function dequeue

   // isQueueEmpty calls List function isEmpty
   bool isQueueEmpty() const 
   {
      return isEmpty(); 
   
   } // end function isQueueEmpty

   // printQueue calls List function print
   void printQueue() const 
   { 
      print(); 
   
   } // end function printQueue

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