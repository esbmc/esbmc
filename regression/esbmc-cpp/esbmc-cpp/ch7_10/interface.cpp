// Fig. 7.22: interface.cpp
// Definition of class Interface
#include "interface.h"       // Interface class definition
#include "implementation.h"  // Implementation class definition

// constructor
Interface::Interface( int v ) 
   : ptr ( new Implementation( v ) )  // initialize ptr
{ 
   // empty body

} // end Interface constructor

// call Implementation's setValue function
void Interface::setValue( int v ) 
{ 
   ptr->setValue( v ); 

} // end function setValue

// call Implementation's getValue function
int Interface::getValue() const 
{ 
   return ptr->getValue(); 

} // end function getValue

// destructor
Interface::~Interface() 
{ 
   delete ptr; 

} // end ~Interface destructor

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
