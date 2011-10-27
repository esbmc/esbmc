// Fig. 6.17: fig06_17.cpp
// Demonstrating the order in which constructors and
// destructors are called.
#include <iostream>

using std::cout;
using std::endl;

// include CreateAndDestroy class definition from create.h
#include "create.h"

void create( void );   // prototype

// global object
CreateAndDestroy first( 1, "(global before main)" );

int main()
{
   cout << "\nMAIN FUNCTION: EXECUTION BEGINS" << endl;

   CreateAndDestroy second( 2, "(local automatic in main)" );

   static CreateAndDestroy third( 
      3, "(local static in main)" );
   
   create();  // call function to create objects

   cout << "\nMAIN FUNCTION: EXECUTION RESUMES" << endl;

   CreateAndDestroy fourth( 4, "(local automatic in main)" );
   
   cout << "\nMAIN FUNCTION: EXECUTION ENDS" << endl;

   return 0;

} // end main

// function to create objects
void create( void )
{
   cout << "\nCREATE FUNCTION: EXECUTION BEGINS" << endl;

   CreateAndDestroy fifth( 5, "(local automatic in create)" );

   static CreateAndDestroy sixth( 
      6, "(local static in create)" );

   CreateAndDestroy seventh( 
      7, "(local automatic in create)" );

   cout << "\nCREATE FUNCTION: EXECUTION ENDS" << endl;

} // end function create

/**************************************************************************
 * (C) Copyright 1992-2002 by Deitel & Associates, Inc. and Prentice      *
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
