/*******************************************************************\

Module: Extracting Counterexamples

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TRANS_COUNTEREXAMPLE_H
#define CPROVER_TRANS_COUNTEREXAMPLE_H

#include <hash_cont.h>
#include <ui_message.h>
#include <threeval.h>
#include <namespace.h>
#include <decision_procedure.h>

#include <solvers/prop/literal.h>

#include "bmc_map.h"

class trans_tracet
{
public:
  class statet
  {
  public:
    class assignmentt
    {
    public:
      exprt lhs;
      exprt rhs;
      locationt location;
      
      assignmentt():location(static_cast<const locationt &>(get_nil_irep()))
      {
      }
    };

    typedef std::list<assignmentt> assignmentst;
    assignmentst assignments;
  };

  // one state per timeframe
  typedef std::vector<statet> statest;
  statest states;
  
  // mode of whole trace
  std::string mode;
  
  // properties
  struct propertyt
  {
    tvt status;
    unsigned failing_timeframe;
  };
  
  typedef std::list<propertyt> propertiest;
  propertiest properties;
  
  unsigned get_failing_timeframe() const
  {
    assert(!states.empty());
    if(properties.empty()) return states.size()-1;
    
    unsigned max=0;
    
    for(propertiest::const_iterator
        it=properties.begin();
        it!=properties.end();
        it++)
    {
      if(it->status.is_false() &&
         it->failing_timeframe>max)
        max=it->failing_timeframe;
    }
    
    return max;
  }
};

void compute_trans_trace(
  const decision_proceduret &decision_procedure,
  unsigned no_timeframes,
  const namespacet &ns,
  const irep_idt &module,
  trans_tracet &dest);
  
void compute_trans_trace(
  const std::list<exprt> &properties,
  const std::list<bvt> &prop_bv,
  const class prop_convt &solver,
  unsigned no_timeframes,
  const namespacet &ns,
  const irep_idt &module,
  trans_tracet &dest);
  
void compute_trans_trace(
  const std::list<bvt> &prop_bv,
  const bmc_mapt &bmc_map,
  const class propt &solver,
  const namespacet &ns,
  trans_tracet &dest);
  
void show_trans_trace(
  const trans_tracet &trace,
  messaget &message,
  const namespacet &ns,
  ui_message_handlert::uit ui);  

void show_trans_trace_vcd(
  const trans_tracet &trace,
  messaget &message,
  const namespacet &ns,
  ui_message_handlert::uit ui,
  std::ostream &out);

#endif
