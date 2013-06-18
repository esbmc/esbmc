// Fig. 21.25: fig21_25.cpp
// Standard library adapter priority_queue test program.
#include <iostream>

using std::cout;
using std::endl;

#include <queue>  // priority_queue adapter definition

int main()
{
   std::priority_queue< double > priorities;
   
   // push elements onto priorities
   priorities.push( 3.2 );
   priorities.push( 9.8 );
   priorities.push( 5.4 );

	cout << "Popping from priorities: ";
   
   while ( !priorities.empty() ) {
      cout << priorities.top() << ' ';  // view top element
      priorities.pop();                 // remove top element

   } // end while

   cout << endl;
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
