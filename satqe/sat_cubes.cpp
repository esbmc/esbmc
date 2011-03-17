/*******************************************************************\

Module: Satisfiablility Cube Generation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <algorithm>

#include "sat_cubes.h"

//#define DEBUG

/*******************************************************************\

Function: sat_hook

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool sat_hook(CSolver *solver)
{
  ((class sat_cubest *)solver)->found_sat();
  return true; // continue solving
}

/*******************************************************************\

Function: sat_cubest::set_important_variables

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void sat_cubest::set_important_variables(
  const std::vector<unsigned> &_important_variables)
{
  important_variables.clear();

  for(std::vector<unsigned>::const_iterator
      it=_important_variables.begin();
      it!=_important_variables.end(); it++)
    important_variables.insert(*it);

  important_variablesv=_important_variables;
}

/*******************************************************************\

Function: compare_var_stat

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

inline bool compare_var_stat(const pair<vector<CVariable>::iterator,int> &v1,
                             const pair<vector<CVariable>::iterator,int> &v2);

/*******************************************************************\

Function: sat_cubest::update_var_stats

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#if 0
void sat_cubest::update_var_stats(void)
{
  for(unsigned int i=1; i<variables().size(); ++i)
  {
    CVariable &var=variable(i);

    // normal Chaff scoring
    var.score(0) = var.score(0)/2 + var.lits_count(0) - _last_var_lits_count[0][i];
    var.score(1) = var.score(1)/2 + var.lits_count(1) - _last_var_lits_count[1][i];

    #if 0 // decide target state variables last
    if(is_important(i)) 
    {
      var.score(0)=0;
      var.score(1)=0;
    }
    #else
    #if 1 // decide target state variables first
    if(is_important(i))
    {
      var.score(0)+=100000;
      var.score(1)+=100000;
    }
    #endif
    #endif

    // normal Chaff scoring
    _last_var_lits_count[0][i] = var.lits_count(0);
    _last_var_lits_count[1][i] = var.lits_count(1);
    _var_order[i-1] = pair<CVariable * , int>( &var, var.score());
  }

  // from here as in Chaff

  ::stable_sort(_var_order.begin(), _var_order.end(), compare_var_stat);

  for(int i=0; i<_var_order.size(); ++i)
    _var_order[i].first->var_score_pos() = i;

  _max_score_pos = 0;
  
  CSolver::update_var_stats();
}
#endif

/*******************************************************************\

Function: sat_cubest::decide_next_branch

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#if 0
bool sat_cubest::decide_next_branch(void)
{
    ++_stats.num_decisions;
    if (!_implication_queue.empty()) {
	//some hook function did a decision, so skip my own decision making.
	//if the front of implication queue is 0, that means it's finished
	//because var index start from 1, so 2 *vid + sign won't be 0. 
	//else it's a valid decision.
	_max_score_pos = 0; //reset the max_score_position.
	return _implication_queue.front().first;
    }
	
    int s_var = 0;

    for (int i=_max_score_pos; i<_var_order.size(); ++i) {
	CVariable & var = *_var_order[i].first;
	if (var.value()==UNKNOWN) {
	    //move th max score position pointer
	    _max_score_pos = i;
	    //make some randomness happen
	    if (--_params.randomness < _params.base_randomness)
		_params.randomness = _params.base_randomness;

	    int randomness = _params.randomness;
  	    if (randomness >= num_free_variables())
  		randomness = num_free_variables() - 1;
            int skip;
            if(num_free_variables()==0)
              skip=0;
            else
	      skip =random()%(1+randomness);

	    int index = i;
	    while (skip > 0) {
		++index;
		if (_var_order[index].first->value()==UNKNOWN)
		    --skip;
	    }
	    vector<CVariable>::iterator ptr = _var_order[index].first;
	    assert (ptr->value() == UNKNOWN);
	    int sign = ptr->score(0) > ptr->score(1) ? 0 : 1;
	    int var_idx = ptr - variables().begin();

            #ifdef USE_PREVIOUS_ASSIGNMENT
            if(!previous_assignment.empty())
              sign=(previous_assignment[var_idx]&1);
            #endif

            #if 0
            if(is_important(var_idx))
              std::cout << "DECISION ON IMPORTANT VARIABLE: "
                        << var_idx << " SIGN " << sign 
                        << "LEVEL " << dlevel()+1 << std::endl;
            else
              std::cout << "DECISION ON " << var_idx << std::endl;
            #endif

	    s_var = var_idx + var_idx + sign;
	    break;
	}
    }

    if (s_var<2) //no more free vars, solution found,  quit
	return false;
    ++dlevel();
    if (dlevel() > _stats.max_dlevel) _stats.max_dlevel = dlevel();
    CHECK (cout << "**Decision at Level " << dlevel() ;
	   cout <<": " << s_var << "\te.g. " << (s_var&0x1?"-":" ") ;
	   cout <<(s_var>>1)  << endl; );
    queue_implication(s_var, NULL_CLAUSE);
    return true;
}
#endif

/*******************************************************************\

Function: sat_cubest::store_assignment

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void sat_cubest::store_assignment()
{
  #ifdef USE_PREVIOUS_ASSIGNMENT
  previous_assignment.resize(variables().size());
  for(unsigned int i=0; i<variables().size(); i++)
    previous_assignment[i]=variable(i).value();
  #endif
}

/*******************************************************************\

Function: sat_cubest::show_assignment

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void sat_cubest::show_assignment()
{
  std::cout << "ASSIGNMENT: " << std::endl;
  for(std::vector<unsigned>::const_iterator
      it=important_variablesv.begin();
      it!=important_variablesv.end(); it++)
  {
    int i=*it;
    switch(variable(i).value()) {
     case DONT_CARE: std::cout << "(*" << i << "*) "; break;
          case 0: std::cout << "-" << i << " "; break;
          case 1: std::cout << i << " "; break;
          default: std::cerr << "Unknown variable value state" << std::endl; exit(1);
    }
  }

  std::cout << std::endl;

  //dump_assignment_stack();
}

/*******************************************************************\

Function: sat_cubest::found_sat

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void sat_cubest::found_sat()
{
  // clear stars
  stars.clear();

  // store assignment
  store_assignment();

  // maybe print assignment  
  #ifdef DEBUG
  show_assignment();
  #endif

  // enlarge
  enlarge();

  // build blocking clause
  add_blocking_clause();
}

/*******************************************************************\

Function: sat_cubest::enlarge

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void sat_cubest::enlarge()
{
  for(std::vector<unsigned>::const_iterator
      it=important_variablesv.begin();
      it!=important_variablesv.end(); it++)
  {
    unsigned int index=*it;

    if(not_used.find(index)!=not_used.end())
      stars.insert(index); 
  }
}

/*******************************************************************\

Function: sat_cubest::add_blocking_clause

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void sat_cubest::add_blocking_clause()
{
  std::vector<int> blocking_clause;

  blocking_clause.reserve(important_variablesv.size());

  cube_sett::bitvt bits, stars_v;
  bits.reserve(important_variablesv.size());
  stars_v.reserve(important_variablesv.size());

  for(std::vector<unsigned>::const_iterator
      it=important_variablesv.begin();
      it!=important_variablesv.end(); it++)
  {
    unsigned int index=*it;

    if(stars.find(index)!=stars.end())
      stars_v.push_back(1);
    else
      switch(variable(index).value())
      {
       case 0: blocking_clause.push_back(2*index);   stars_v.push_back(0); bits.push_back(0); break;
       case 1: blocking_clause.push_back(2*index+1); stars_v.push_back(0); bits.push_back(1); break;
       default: stars_v.push_back(1); break;
      }
  }

  // show some progress
  {
    static unsigned cnt=0;
    cnt++;
    if((cnt&0x1ff)==0x100)
      std::cout << "SIZE: " << cnt << std::endl;
  }

  #ifdef USE_CUBE_SET
  if(cube_set1!=NULL)
    cube_set1->insert(stars_v, bits);

  if(cube_set2!=NULL)
    cube_set2->insert(stars_v, bits);
  #endif

  #ifdef DEBUG
  std::cout << "Blocking clause: ";
  for(unsigned i=0; i<blocking_clause.size(); i++)
    std::cout << " " 
              << ((blocking_clause[i]&1)?"-":"")
              << (blocking_clause[i] >> 1);
  std::cout << "\n";
  #endif

  if(cube_list!=NULL)
    cube_list->push_back(blocking_clause);

  if(blocking_clause.size()==0)
  {
    // DONE
    _stats.outcome = UNSATISFIABLE;
  }
  else if(add_blocking_clause(blocking_clause)<0)
  {
    _stats.outcome = UNSATISFIABLE;
    // done, unsat
    //std::cout << "DONE, UNSAT\n";
  }
}

/*******************************************************************\

Function: sat_cubest::add_blocking_clause

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int sat_cubest::add_blocking_clause(
  std::vector<int> &blocking_clause)
{
  int *lits=&blocking_clause.front();

  ClauseIdx added_cl=
    CSolver::add_orig_clause(lits, blocking_clause.size());

  if(added_cl<0) //memory out.
    abort();
  
  //adjust_variable_order(lits, blocking_clause.size());

  int back_dl=0;
    
  for(unsigned i=0; i<clause(added_cl).num_lits(); ++i)
  {
    int vid=clause(added_cl).literal(i).var_index();
    //int sign=clause(added_cl).literal(i).var_sign();
    assert(variable(vid).value()!=UNKNOWN);
    assert(literal_value(clause(added_cl).literal(i))==0);
    int dl=variable(vid).dlevel();

    //std::cout << "DL: " << dl << " " << dlevel() << std::endl;

    back_dl=std::max(dl, back_dl);
  }

  //std::cout << "BDL: " << back_dl << " " << dlevel() << std::endl;

  if(back_dl<dlevel())
    back_track(back_dl+1);

  _conflicts.push_back(added_cl);
  return analyze_conflicts();
}

/*******************************************************************\

Function: sat_cubest::preprocess

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int sat_cubest::preprocess(void) 
{
  not_used.clear();

  for(int i=1, sz=variables().size(); i<sz; ++i)
  {
    CVariable &v=variable(i);

    if(v.lits_count(0)==0 && v.lits_count(1)==0)
      not_used.insert(i);
  }

  int r=CSolver::preprocess();

  return r;
}
