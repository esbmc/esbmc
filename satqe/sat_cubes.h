/*******************************************************************\

Module: Satisfiablility Cube Generation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_SAT_CUBES_H
#define CPROVER_SAT_CUBES_H

// options

// #define USE_PREVIOUS_ASSIGNMENT
#define USE_CUBE_SET

#include <list>
#include <map>
#include <set>

#include <zchaff_solver.h>

#ifdef USE_CUBE_SET
#include "cube_set.h"
#endif

#define DONT_CARE       3

#if 0
class varct
{
 public:
  typedef std::list<int> clause_listt;
  clause_listt clause_list;
};
#endif

bool sat_hook(CSolver *solver);

class sat_cubest:public CSolver
{
public:
  sat_cubest(const CSolver &_solver):CSolver(_solver)
  { 
    init();
  }
   
  sat_cubest() { init(); }
  
  virtual ~sat_cubest() { }

  void set_important_variables(const std::vector<unsigned> &_important_variables);

  typedef std::list<std::vector<int> > cube_listt;
  cube_listt *cube_list;

  cube_sett *cube_set1;
  cube_sett *cube_set2;

  virtual void found_sat();   

protected:
  // chaff overloads
  virtual int preprocess();
  //virtual bool decide_next_branch(void);
  //virtual void update_var_stats(void);
  
  int add_blocking_clause(std::vector<int> &clause);

  std::set<int> not_used;

protected:
  // new methods
  void show_assignment();
  void store_assignment();
  virtual void add_blocking_clause();
  virtual void enlarge();

  typedef std::set<unsigned> important_variablest;  
  important_variablest important_variables;
  std::vector<unsigned> important_variablesv;

  std::set<unsigned> stars;
  
  #ifdef USE_PREVIOUS_ASSIGNMENT
  std::vector<char> previous_assignment;
  #endif

protected:
  void init()
  {
    cube_list=NULL;
    cube_set1=cube_set2=NULL;
    add_sat_hook(sat_hook);
  }
 
  bool is_important(unsigned int i)
  {
    return important_variables.find(i)!=
           important_variables.end();
  }
};

#endif
