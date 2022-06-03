// Fig. 5.9: GradeBook.h
// GradeBook class definition that counts letter grades.
// Member functions are defined in GradeBook.cpp
#include <string> // program uses C++ standard string class

// GradeBook class definition
class GradeBook
{
public:
   explicit GradeBook( std::string ); // initialize course name
   void setCourseName( std::string ); // set the course name
   std::string getCourseName() const; // retrieve the course name
   void displayMessage() const; // display a welcome message
   void inputGrades(); // input arbitrary number of grades from user
   void displayGradeReport() const;  // display report based on user input
private:
   std::string courseName; // course name for this GradeBook
   unsigned int aCount; // count of A grades
   unsigned int bCount; // count of B grades
   unsigned int cCount; // count of C grades
   unsigned int dCount; // count of D grades
   unsigned int fCount; // count of F grades
   bool flag = false;   // C++11 in-class initializers
}; // end class GradeBook

/**************************************************************************
 * (C) Copyright 1992-2014 by Deitel & Associates, Inc. and               *
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
