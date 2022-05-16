// Fig. 3.17: fig03_16.cpp
// Create and manipulate a GradeBook object; illustrate validation.
#include <iostream>
#include "GradeBook.h" // include definition of class GradeBook
using namespace std;

// function main begins program execution
int main()
{
   // create two GradeBook objects; 
   // initial course name of gradeBook1 is too long
   GradeBook gradeBook1( "CS101 Introduction to Programming in C++" );
   GradeBook gradeBook2( "CS102 C++ Data Structures" );

   // display each GradeBook's courseName 
   cout << "gradeBook1's initial course name is: " 
      << gradeBook1.getCourseName()
      << "\ngradeBook2's initial course name is: " 
      << gradeBook2.getCourseName() << endl;

   // modify gradeBook1's courseName (with a valid-length string)
   gradeBook1.setCourseName( "CS101 C++ Programming" );

   // display each GradeBook's courseName 
   cout << "\ngradeBook1's course name is: " 
      << gradeBook1.getCourseName()
      << "\ngradeBook2's course name is: " 
      << gradeBook2.getCourseName() << endl;
} // end main



/**************************************************************************
 * (C) Copyright 1992-2012 by Deitel & Associates, Inc. and               *
 * Pearson Education, Inc. All Rights Reserved.                           *
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
 **************************************************************************/
