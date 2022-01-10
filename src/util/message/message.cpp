/*******************************************************************\

Module: 

Author: Rafael Menezes, rafael.sa.menezes@outlook.com

Maintainers:
\*******************************************************************/

#include <util/message/message.h>
#include <util/filesystem.h>

file_operations::tmp_file messaget::get_temp_file()
{
  return file_operations::create_tmp_file("esbmc-%%%%-%%%%");
}

void messaget::insert_file_contents(VerbosityLevel l, FILE *f) const
{
  const int MAX_LINE_LENGTH = 1024;
  // Go to the beginning of the file
  fseek(f, 0, SEEK_SET);
  char line[MAX_LINE_LENGTH] = {0};

  // Read every line of file
  while(fgets(line, MAX_LINE_LENGTH, f))
    // Send to message print method
    print(l, line);
}
