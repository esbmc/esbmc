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

protected:
  virtual std::string convert_to_string() = 0;

private:
  T expression;
};

/**
 *  A generic interface to represent a container to store SSA steps
 */
template <ssa_container_item T>
class ssa_container
{
public:
  virtual T get()
  {
    return expressions;
  }
  explicit expr_container(const T &other) : expressions(other)
  {
  }

private:
  T expressions;
};

template <ssa_container_item T>
class ssa_set_container : public ssa_container<std::set<T>>
{
  explicit ssa_set_container(const std::set<T> &other) : ssa_container(other)
  {
  }
};

#endif //ESBMC_SSA_CONTAINER_H
