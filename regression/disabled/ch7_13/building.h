// Fig. 7.25: building.h
// Building class definition.
#ifndef BUILDING_H
#define BUILDING_H

#include "elevator.h"   // Elevator class definition
#include "floor.h"      // Floor class definition
#include "clock.h"      // Clock class definition
#include "scheduler.h"  // Scheduler class definition

class Building {

public:
   Building();                 // constructor
   ~Building();                // destructor
   void runSimulation( int );  // controls simulation 

private:
   Floor floor1;               // floor1 object
   Floor floor2;               // floor2 object
   Elevator elevator;          // elevator object
   Clock clock;                // clock object
   Scheduler scheduler;        // scheduler object

}; // end class Building

#endif // BUILDING_H

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
