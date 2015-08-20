// Fig. 16.8: environment.cpp
// Program to display CGI environment variables.
#include <iostream>

using std::cout;

#include <string>

using std::string;

#include <cstdlib>

int main()
{
   string environmentVariables[ 24 ] = {
      "COMSPEC", "DOCUMENT_ROOT", "GATEWAY_INTERFACE",
      "HTTP_ACCEPT", "HTTP_ACCEPT_ENCODING",
      "HTTP_ACCEPT_LANGUAGE", "HTTP_CONNECTION", 
      "HTTP_HOST", "HTTP_USER_AGENT", "PATH", 
      "QUERY_STRING", "REMOTE_ADDR", "REMOTE_PORT",
      "REQUEST_METHOD", "REQUEST_URI", "SCRIPT_FILENAME",
      "SCRIPT_NAME", "SERVER_ADDR", "SERVER_ADMIN",
      "SERVER_NAME","SERVER_PORT","SERVER_PROTOCOL",
      "SERVER_SIGNATURE","SERVER_SOFTWARE" };
   
   // output header
   cout << "Content-type: text/html\n\n";
   
   // output XML declaration and DOCTYPE
   cout << "<?xml version = \"1.0\"?>"
        << "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 "
        << "Transitional//EN\" \"http://www.w3.org/TR/xhtml1"
        << "/DTD/xhtml1-transitional.dtd\">";

   // output html element and some of its contents
   cout << "<html xmlns = \"http://www.w3.org/1999/xhtml\">"
        << "<head><title>Environment Variables</title></head>"
        << "<body>";
   
   // begin outputting table
   cout << "<table border = \"0\" cellspacing = \"2\">";
   
   // iterate through environment variables
   for ( int i = 0; i < 24; i++ )
      cout << "<tr><td>" << environmentVariables[ i ] 
           << "</td><td>"
           << getenv( environmentVariables[ i ].data() )
           << "</td></tr>";
   
   cout << "</table></body></html>";
   
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
