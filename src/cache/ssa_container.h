/*******************************************************************\
 Module: SSA Container

 Author: Rafael Sá Menezes

 Date: April 2020

 Description:
  This file will define methods and classes to help manipulate stored
  SSA's as the cache will work in SSA, this a form of storing and
  manipulate it, this may be a simple set with custom hashing
  to tries and external databases.
\*******************************************************************/

#ifndef ESBMC_SSA_CONTAINER_H
#define ESBMC_SSA_CONTAINER_H

#include <string>
/**
 *  A generic interface to represent a item to stored in the SSA container
 *  this is the form that the object will be represented in the storage system
 *  e.g hash, string, custom object
 */
template <class T>
class ssa_container_item
{
public:
  explicit ssa_container_item(const T &other) : expression(other)
  {
  }
  virtual T get()
  {
    return expression;
  }

  virtual void operator=(const ssa_container_item<T> &other)
  {
    expression = other.expression;
  }
  virtual void set(const ssa_container_item<T> &other)
  {
    expression = other.expression;
  }

protected:
  T expression;
};

/**
 *  A generic interface to represent a container to store SSA steps
 */
template <class T>
class ssa_container
{
public:
  virtual T get()
  {
    return expressions;
  }

  explicit ssa_container() : expressions()
  {
  }

  virtual void operator=(const ssa_container<T> &other)
  {
    expressions = other.expressions;
  }

  virtual void set(const ssa_container<T> &other)
  {
    expressions = other.expressions;
  }

protected:
  T expressions;
};

template <class T>
class container_storage
{
public:
  container_storage() = default;
  virtual ssa_container<T> load(std::istream &) = 0;
  virtual void store(ssa_container<T> &) = 0;
};

#endif //ESBMC_SSA_CONTAINER_H
