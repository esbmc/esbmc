// Fig. 6.16: create.cpp
// Member-function definitions for class CreateAndDestroy
#include <iostream>

using std::cout;
using std::endl;

// include CreateAndDestroy class definition from create.h
#include "create.h"

// constructor
CreateAndDestroy::CreateAndDestroy( 
   int objectNumber, char *messagePtr )
{
   objectID = objectNumber;
   message = messagePtr;

   cout << "Object " << objectID << "   constructor runs   "
        << message << endl;

} // end CreateAndDestroy constructor

// destructor
CreateAndDestroy::~CreateAndDestroy()
{ 
   // the following line is for pedagogic purposes only
   cout << ( objectID == 1 || objectID == 6 ? "\n" : "" );

   cout << "Object " << objectID << "   destructor runs    " 
        << message << endl; 

} // end ~CreateAndDestroy destructor

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
