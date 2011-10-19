// Fig. 6.4: fig06_04.cpp
// Demonstrating the class member access operators . and ->
//
// CAUTION: IN FUTURE EXAMPLES WE AVOID PUBLIC DATA!
#include <iostream>

using std::cout;
using std::endl;

// class Count definition
class Count {

public:
   int x;

   void print() 
   { 
      cout << x << endl; 
   }

}; // end class Count

int main()
{
   Count counter;                // create counter object 
   Count *counterPtr = &counter; // create pointer to counter
   Count &counterRef = counter;  // create reference to counter

   cout << "Assign 1 to x and print using the object's name: ";
   counter.x = 1;       // assign 1 to data member x
   counter.print();     // call member function print

   cout << "Assign 2 to x and print using a reference: ";
   counterRef.x = 2;    // assign 2 to data member x
   counterRef.print();  // call member function print

   cout << "Assign 3 to x and print using a pointer: ";
   counterPtr->x = 3;   // assign 3 to data member x
   counterPtr->print(); // call member function print

   return 0;  

} // end main

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