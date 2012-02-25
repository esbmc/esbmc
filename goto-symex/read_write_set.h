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

class read_write_set {
public:
    read_write_set()
    {

    };

    virtual ~read_write_set();
    std::set<irep_idt> read_set;
    std::set<irep_idt> write_set;

    bool empty() const
    {
        return read_set.empty() && write_set.empty();
    }

    bool has_write_intersect(const std::set<irep_idt> & _set) const
    {
        if(write_set.empty() || _set.empty())
          return false;

        for(std::set<irep_idt>::iterator it1 = write_set.begin(); it1 != write_set.end(); it1++)
        {
            for(std::set<irep_idt>::iterator it2 = _set.begin(); it2 != _set.end(); it2++)
            {
                if(*it1 == *it2)
                  return true;
            }
        }
        return false;
    }

    bool has_read_intersect(const std::set<irep_idt> & _set) const
    {

        if(read_set.empty() || _set.empty())
          return false;

        for(std::set<irep_idt>::iterator it1 = read_set.begin(); it1 != read_set.end(); it1++)
        {
            for(std::set<irep_idt>::iterator it2 = _set.begin(); it2 != _set.end(); it2++)
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

