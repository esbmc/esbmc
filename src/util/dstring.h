/*******************************************************************\

Module: Container for C-Strings

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef DSTRING_H
#define DSTRING_H

#include <iostream>
#include <util/string_container.h>

class dstring final
{
public:
  // this is safe for static objects

  constexpr dstring() : no(0)
  {
  }

  // this is safe for static objects
  constexpr static dstring make_from_table_index(unsigned no)
  {
    return dstring(no);
  }

  // this one is not safe for static objects
  dstring(const char *s) : no(get_string_container()[s])
  {
  }

  // this one is not safe for static objects
  dstring(const std::string &s) : no(get_string_container()[s])
  {
  }

  dstring(const dstring &) = default;

  // access
  /// source object \p other, this is effectively just a copy constructor.
  constexpr dstring(dstring &&other) : no(other.no)
  {
  }

  inline bool empty() const
  {
    return no == 0; // string 0 is exactly the empty string
  }

  inline char operator[](size_t i) const
  {
    return as_string()[i];
  }

  // warning! the address returned is not stable
  inline const char *c_str() const
  {
    return as_string().c_str();
  }

  inline size_t size() const
  {
    return as_string().size();
  }

  // ordering -- not the same as lexicographical ordering

  inline bool operator<(const dstring &b) const
  {
    return no < b.no;
  }

  // comparison with same type

  inline bool operator==(const dstring &b) const
  {
    return no == b.no;
  } // really fast equality testing

  inline bool operator!=(const dstring &b) const
  {
    return no != b.no;
  } // really fast equality testing

  // comparison with other types

  bool operator==(const char *b) const
  {
    return as_string() == b;
  }
  bool operator!=(const char *b) const
  {
    return as_string() != b;
  }

  bool operator==(const std::string &b) const
  {
    return as_string() == b;
  }
  bool operator!=(const std::string &b) const
  {
    return as_string() != b;
  }
  bool operator<(const std::string &b) const
  {
    return as_string() < b;
  }
  bool operator>(const std::string &b) const
  {
    return as_string() > b;
  }
  bool operator<=(const std::string &b) const
  {
    return as_string() <= b;
  }
  bool operator>=(const std::string &b) const
  {
    return as_string() >= b;
  }

  int compare(const dstring &b) const
  {
    if(no == b.no)
      return 0; // equal
    return as_string().compare(b.as_string());
  }

  // the reference returned is guaranteed to be stable
  const std::string &as_string() const
  {
    return get_string_container().get_string(no);
  }

  // modifying

  inline void clear()
  {
    no = 0;
  }

  inline void swap(dstring &b)
  {
    unsigned t = no;
    no = b.no;
    b.no = t;
  }

  inline dstring &operator=(const dstring &b) = default;
  inline dstring &operator=(dstring &&other) = default;

  // output

  std::ostream &operator<<(std::ostream &out) const
  {
    return out << as_string();
  }

  inline unsigned get_no() const
  {
    return no;
  }

  inline size_t hash() const
  {
    return no;
  }

private:
  constexpr explicit dstring(unsigned _no) : no(_no)
  {
  }

  unsigned no;
};

struct dstring_hash
{
  size_t operator()(const dstring &s) const
  {
    return s.hash();
  }
};

inline std::ostream &operator<<(std::ostream &out, const dstring &a)
{
  return a.operator<<(out);
}

#endif
