/*
 * File:   read_write_set.h
 * Author: yspb1g08
 *
 * Created on 19 September 2009, 20:54
 */

#ifndef _READ_WRITE_SET_H
#define	_READ_WRITE_SET_H

#include <set>
#include <std_expr.h>
#include <string_hash.h>

class read_write_set {
public:
    read_write_set()
    {

    };

    read_write_set(const read_write_set& orig);
    virtual ~read_write_set();
    typedef hash_set_cont<string_wrapper, string_wrap_hash> string_set;
    string_set read_set;
    string_set write_set;

    bool empty() const
    {
        return read_set.empty() && write_set.empty();
    }

    bool has_write_intersect(const string_set& _set) const
    {
        if(write_set.empty() || _set.empty())
          return false;

        for (string_set::const_iterator it1 = write_set.begin(); it1 != write_set.end(); it1++)
        {
            for (string_set::const_iterator it2 = _set.begin(); it2 != _set.end(); it2++)
            {
                if(*it1 == *it2)
                  return true;
            }
        }
        return false;
    }

    bool has_read_intersect(const string_set & _set) const
    {

        if(read_set.empty() || _set.empty())
          return false;

        for (string_set::const_iterator it1 = read_set.begin(); it1 != read_set.end(); it1++)
        {
            for (string_set::const_iterator it2 = _set.begin(); it2 != _set.end(); it2++)
            {
                if(*it1 == *it2)
                  return true;
            }
        }
        return false;
    }
private:

};

#endif	/* _READ_WRITE_SET_H */

