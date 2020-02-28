/*******************************************************************\

Module: binary irep conversions with hashing

Author: CM Wintersteiger

Date: May 2007

\*******************************************************************/

#ifndef IREP_SERIALIZATION_H_
#define IREP_SERIALIZATION_H_

#include <map>
#include <util/irep2.h>
#include "irep2_type.h"

/**
 * Class used for irep serialization, containing methods to save/load for later
 * usage.
 */
class irep_serializationt
{
private:
  /**
   * Helper class to generate and compare irep hashs
   */
  struct irep_full_hash
  {
    /**
     * Takes an irep and generates it's hash
     * @param i irep to be hashed
     * @return the hash value of i
     */
    size_t operator()(const irept &i) const
    {
      return i.full_hash();
    }

    /**
     * Compares two irep using their hashes
     * @param i1 first irep
     * @param i2 second irep
     * @return truth value for if i1 hash is lower than i2 hash
     */
    bool operator()(const irept &i1, const irept &i2) const
    {
      return i1.full_hash() < i2.full_hash();
    }
  };

public:
  /**
   * Helper class to keep track of which ireps where read or wrote.
   */
  class ireps_containert
  {
  public:
    typedef std::map<unsigned, irept> irepts_on_readt;
    irepts_on_readt ireps_on_read;

    typedef std::vector<irept> irepts_on_writet;
    irepts_on_writet ireps_on_write;

    typedef std::vector<bool> string_mapt;
    string_mapt string_map;

    typedef std::vector<std::pair<bool, dstring>> string_rev_mapt;
    string_rev_mapt string_rev_map;

    /**
     * Removes entries from all dictionaries and vectors
     */
    void clear()
    {
      ireps_on_write.clear();
      ireps_on_read.clear();
      string_map.clear();
      string_rev_map.clear();
    }
  };

  /**
   * Constructor for serialization
   * @param ic container which will hold the serialization.
   */
  explicit irep_serializationt(ireps_containert &ic) : ireps_container(ic){};

  /**
   * Checks if a reference was already read and then stores it in an irept
   * reference
   * @param in input containing the irep
   * @param irep object where the irep will be stored
   */
  void reference_convert(std::istream &in, irept &irep);

  /**
 * Checks if a reference was already written if yes, then stores it's id
 * if not, then stores it's id alongside with irep
 * @param irep irept to be stored
 * @param out output stream
 */
  void reference_convert(const irept &irep, std::ostream &out);

  /**
   * Read a string reference from input stream and adds it into string_rev_map
   * @param in input stream to be read
   * @return a dstring with the input
   */
  irep_idt read_string_ref(std::istream &in);

  /**
   * Write an dstring reference into an output buffer and adds it to the
   * string_map
   * @param out output buffer
   * @param s dstring to be wrote
   */
  void write_string_ref(std::ostream &out, const dstring &s);

  /**
   * Clear current container
   */
  void clear()
  {
    ireps_container.clear();
  }

  /**
   * Reads an input as an 4 byte unsigned
   * @param in input stream
   * @return 4-byte unsigned with the value read
   */
  static unsigned read_long(std::istream &in);

  /**
 * Read a dstring from a buffer
 * @param in
 * @return
 */
  static dstring read_string(std::istream &in);

  /**
   * Sends a long to a output stream
   * @param out output stream
   * @param u long to be sent to output
   */
  static void write_long(std::ostream &out, unsigned u);

  /**
 * Sends a string to a output stream
 * @param out output stream
 * @param s string to be sent to output
 */
  static void write_string(std::ostream &out, const std::string &s);

  /**
   * Since the irep2 does not inherit for a common type an union type will be
   * used
   */
  union irep2tc {
    std::unique_ptr<type2tc> t;
    std::unique_ptr<expr2tc> e;
  };
  struct irep2_unserialization
  {
    bool is_expr;
    union irep2tc c;
  };
  /**
   * Writes an irep2 to an output stream
   * @param out output stream
   * @param irep2 to be serialized
   */
  template <class T>
  static void write_irep2(std::ostream &out, const T &irep2);

  /**
   * Read an input stream and converts it to an irept
   * @param in input stream to be read
   * @param irep reference to be initizalized
   */
  static void read_irep(std::istream &in, irept &irep);

private:
  ireps_containert &ireps_container;

  /**
   * Writes an irep to an output stream
   * @param out output stream
   * @param irep to be serialized
   */
  void write_irep(std::ostream &out, const irept &irep);

  /**
   * Read an input stream and converts it to an irept
   * @param in input stream to be read
   * @param irep reference to be initizalized
   */
  void read_irep_helper(std::istream &in, irept &irep);
};

#endif /*IREP_SERIALIZATION_H_*/
