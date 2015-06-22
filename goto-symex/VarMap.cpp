// Bruno J. Savino
#include <iostream>

#include <cstring>
#include "VarMap.h"


// default VarMap constructor
VarMap::VarMap( std::string lineNumber,
		std::string lineInfo,
		std::string varName,
      std::string varInfo )
{
   setLineNumber( lineNumber );
   setLineInfo( lineInfo );
   setVarName( varName );
   setVarInfo( varInfo );
} // end VarMap constructor

std::string VarMap::getLineNumber() const
{
   return lineNumber;
} 

void VarMap::setLineNumber( std::string lineNumberString )
{

   const char *lineNumberValue = lineNumberString.data();
   int length = strlen( lineNumberValue );
   length = ( length < LNUMBERSIZE ? length : LNUMBERSIZE-1 );
   strncpy( lineNumber, lineNumberValue, length );


   lineNumber[ length ] = '\0';

} 

std::string VarMap::getLineInfo() const
{
   return lineInfo;
}

void VarMap::setLineInfo( std::string lineInfoString )
{

   const char *lineInfoValue = lineInfoString.data();
   int length = strlen( lineInfoValue );
   length = ( length < LINFOSIZE ? length : LINFOSIZE-1 );
   strncpy( lineInfo, lineInfoValue, length );


   lineInfo[ length ] = '\0';
}

std::string VarMap::getVarName() const
{
   return varName;
}

void VarMap::setVarName( std::string varNameString )
{

   const char *varNameValue = varNameString.data();
   int length = strlen( varNameValue );
   length = ( length < VNAMESIZE ? length : VNAMESIZE-1 );
   strncpy( varName, varNameValue, length );

   varName[ length ] = '\0';

}

std::string VarMap::getVarInfo() const
{
   return varInfo;
} 

void VarMap::setVarInfo( std::string varInfoString )
{
   
   const char *varInfoValue = varInfoString.data();
   int length = strlen( varInfoValue );
   length = ( length < VINFOSIZE ? length : VINFOSIZE-1 );
   strncpy( varInfo, varInfoValue, length );


   varInfo[ length ] = '\0';
}
