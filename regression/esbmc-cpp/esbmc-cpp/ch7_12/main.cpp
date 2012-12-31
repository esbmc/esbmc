// Fig. 7.24: elevatorSimulation.cpp
// Driver for the simulation.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

#include "building.h"  // Building class definition

int main()
{
   int duration;  // length of simulation in seconds

   cout << "Enter run time: "; 
   cin >> duration;
   cin.ignore();       // ignore return char
   
   Building building;  // create the building
      
   cout << endl << "*** ELEVATOR SIMULATION BEGINS ***" 
        << endl << endl;

   building.runSimulation( duration );  // start simulation

   cout << "*** ELEVATOR SIMULATION ENDS ***" << endl;

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
