// Fig. 17.14: fig17_14.cpp
// Template Queue class test program.
#include <iostream>

using std::endl;

#include "queue.h"  // Queue class definition

int main()
{
   Queue< int > intQueue;  // create Queue of ints

   cout << "processing an integer Queue" << endl;

   // enqueue integers onto intQueue
   for ( int i = 0; i < 4; i++ ) {
      intQueue.enqueue( i );
      intQueue.printQueue();

   } // end for

   // dequeue integers from intQueue
   int dequeueInteger;

   while ( !intQueue.isQueueEmpty() ) {
      intQueue.dequeue( dequeueInteger );
      cout << dequeueInteger << " dequeued" << endl;
      intQueue.printQueue();

   } // end while

   Queue< double > doubleQueue;  // create Queue of doubles
   double value = 1.1;

   cout << "processing a double Queue" << endl;

   // enqueue floating-point values onto doubleQueue
   for ( int j = 0; j< 4; j++ ) {
      doubleQueue.enqueue( value );
      doubleQueue.printQueue();
      value += 1.1;

   } // end for

   // dequeue floating-point values from doubleQueue
   double dequeueDouble;

   while ( !doubleQueue.isQueueEmpty() ) {
      doubleQueue.dequeue( dequeueDouble );
      cout << dequeueDouble << " dequeued" << endl;
      doubleQueue.printQueue();

   } // end while

   return 0;

} // end main

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