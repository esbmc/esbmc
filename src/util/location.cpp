#include <util/location.h>
#include <util/crypto_hash.h>
std::string locationt::as_string() const
{
  std::string dest;

  const irep_idt &file = get_file();
  const irep_idt &line = get_line();
  const irep_idt &column = get_column();
  const irep_idt &function = get_function();

  if (file != "")
  {
    if (dest != "")
      dest += " ";
    dest += "file " + id2string(file);
  }
  if (line != "")
  {
    if (dest != "")
      dest += " ";
    dest += "line " + id2string(line);
  }
  if (column != "")
  {
    if (dest != "")
      dest += " ";
    dest += "column " + id2string(column);
  }
  if (function != "")
  {
    if (dest != "")
      dest += " ";
    dest += "function " + id2string(function);
  }

  return dest;
}

std::ostream &operator<<(std::ostream &out, const locationt &location)
{
  if (location.is_nil())
    return out;

  out << location.as_string();

  return out;
}

std::string locationt::sha1() const
{
  crypto_hash hash;
  const int line = atoi(get_line().c_str());
  const int column = atoi(get_column().c_str());
  
  hash.ingest(&line, sizeof(int));  
  hash.ingest(&column, sizeof(int));

  hash.fin();

  return hash.to_string();
}
