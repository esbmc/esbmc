// Fig. 11.3: fig11_03.cpp
// Stack-class-template test program.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

#include "tstack1.h"  // Stack class template definition

int main()
{
   Stack< double > doubleStack( 5 );
   double doubleValue = 1.1;

   cout << "Pushing elements onto doubleStack\n";

   while ( doubleStack.push( doubleValue ) ) { 
      cout << doubleValue << ' ';
      doubleValue += 1.1;

   } // end while

   cout << "\nStack is full. Cannot push " << doubleValue
        << "\n\nPopping elements from doubleStack\n";

   while ( doubleStack.pop( doubleValue ) )
      cout << doubleValue << ' ';

   cout << "\nStack is empty. Cannot pop\n";

   Stack< int > intStack;
   int intValue = 1;
   cout << "\nPushing elements onto intStack\n";

   while ( intStack.push( intValue ) ) {
      cout << intValue << ' ';
      ++intValue;

   } // end while

   cout << "\nStack is full. Cannot push " << intValue 
        << "\n\nPopping elements from intStack\n";

   while ( intStack.pop( intValue ) )  
      cout << intValue << ' ';

   cout << "\nStack is empty. Cannot pop\n";

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