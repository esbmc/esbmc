// Fig. 7.26: building.cpp
// Member-function definitions for class Building.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

#include "building.h"  // Building class definition

// constructor
Building::Building() 
   : floor1( Floor::FLOOR1, elevator ), 
     floor2( Floor::FLOOR2, elevator ),
     elevator( floor1, floor2 ), 
     scheduler( floor1, floor2 )
{ 
   cout << "building constructed" << endl; 

} // end Building constructor

// destructor
Building::~Building() 
{ 
   cout << "building destructed" << endl; 

} // end ~Building destructor

// function to control simulation
void Building::runSimulation( int totalTime )
{
   int currentTime = 0;

   while ( currentTime < totalTime ) {
      clock.tick();                   // increment time  
      currentTime = clock.getTime();  // get new time  
      cout << "TIME: " << currentTime << endl;   
      
      // process person arrivals for currentTime
      scheduler.processTime( currentTime );

      // process elevator events for currentTime
      elevator.processTime( currentTime );

      // wait for Enter key press, so user can view output 
      cin.get(); 

   } // end while

} // end function runSimulation

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
