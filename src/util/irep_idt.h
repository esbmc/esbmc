#ifndef IREP_IDT_H
#define IREP_IDT_H

#include <algorithm>
#include <functional>
#include <util/string_pool.h>
#include <util/prefix.h>

// A 4-byte handle into a global string interning pool. Equality and hashing
// are O(1) (integer compare); the underlying string is fetched on demand.
class irep_idt final
{
public:
  // this is safe for static objects

  constexpr irep_idt() : no(0)
  {
  }

  // this is safe for static objects
  constexpr static irep_idt make_from_table_index(unsigned no)
  {
    return irep_idt(no);
  }

  // this one is not safe for static objects
  irep_idt(const char *s) : no(get_string_pool()[s])
  {
  }

  // this one is not safe for static objects
  irep_idt(const std::string &s) : no(get_string_pool()[s])
  {
  }

  irep_idt(const irep_idt &) = default;

  // access
  /// source object \p other, this is effectively just a copy constructor.
  constexpr irep_idt(irep_idt &&other) : no(other.no)
  {
  }

  friend bool has_prefix(const irep_idt &s, std::string_view prefix)
  {
    return has_prefix(std::string_view(s.as_string()), prefix);
  }

  friend bool has_suffix(const irep_idt &s, std::string_view suffix)
  {
    return has_suffix(std::string_view(s.as_string()), suffix);
  }

  inline bool empty() const
  {
    return no == 0; // string 0 is exactly the empty string
  }

  inline char operator[](size_t i) const
  {
    return as_string()[i];
  }

  // the address returned is guaranteed to be stable
  inline const char *c_str() const
  {
    return as_string().c_str();
  }

  inline size_t size() const
  {
    return as_string().size();
  }

  // ordering -- not the same as lexicographical ordering

  inline bool operator<(const irep_idt &b) const
  {
    return no < b.no;
  }

  // comparison with same type

  inline bool operator==(const irep_idt &b) const
  {
    return no == b.no;
  } // really fast equality testing

  inline bool operator!=(const irep_idt &b) const
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

  int compare(const irep_idt &b) const
  {
    if (no == b.no)
      return 0; // equal
    return as_string().compare(b.as_string());
  }

  // the reference returned is guaranteed to be stable
  const std::string &as_string() const
  {
    return get_string_pool().get_string(no);
  }

  // modifying

  inline void clear()
  {
    no = 0;
  }

  inline void swap(irep_idt &b)
  {
    unsigned t = no;
    no = b.no;
    b.no = t;
  }

  inline irep_idt &operator=(const irep_idt &b) = default;
  inline irep_idt &operator=(irep_idt &&other) = default;

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
  constexpr explicit irep_idt(unsigned _no) : no(_no)
  {
  }

  unsigned no;
};

inline std::ostream &operator<<(std::ostream &out, const irep_idt &a)
{
  return a.operator<<(out);
}

namespace std
{
template <>
struct hash<irep_idt>
{
  size_t operator()(const irep_idt &s) const
  {
    return s.hash();
  }
};
} // namespace std

#endif
