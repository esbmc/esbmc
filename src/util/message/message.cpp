/*******************************************************************\

Module: 

Author: Rafael Menezes, rafael.sa.menezes@outlook.com

Maintainers:
\*******************************************************************/

#include <util/message/message.h>
#include <util/filesystem.h>

FILE *messaget::get_temp_file()
{  
  FILE *f = fopen(file_operations::get_unique_tmp_path("esbmc-%%%%-%%%%").c_str(), "w+");
  return f;
}

void messaget::insert_and_close_file_contents(VerbosityLevel l, FILE *f) const
{
  const int MAX_LINE_LENGTH = 1024;
  // Go to the beginning of the file
  fseek(f, 0, SEEK_SET);
  char line[MAX_LINE_LENGTH] = {0};

  // Read every line of file
  while(fgets(line, MAX_LINE_LENGTH, f))
    // Send to message print method
    print(l, line);

  // Close the file
  fclose(f);
}
