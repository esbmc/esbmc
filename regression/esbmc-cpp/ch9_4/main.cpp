// Fig. 9.29: fig09_29.cpp
// Display order in which base-class and derived-class 
// constructors are called.
#include <iostream>  

using std::cout;
using std::endl;

#include "circle5.h"  // Circle5 class definition

int main()
{
   { // begin new scope

      Point4 point( 11, 22 );
   
   } // end scope

   cout << endl;
   Circle5 circle1( 72, 29, 4.5 ); 

   cout << endl;
   Circle5 circle2( 5, 5, 10 ); 

   cout << endl;

   return 0;  // indicates successful termination
   
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