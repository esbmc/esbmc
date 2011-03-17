/*******************************************************************\

Module: SMT-LIB Frontend, Typechecking

Author: CM Wintersteiger

\*******************************************************************/

#include <algorithm>
#include <i2string.h>
#include <assert.h>

#include "smt_typecheck.h"
#include "smt_finalizer_generic.h"
#include "smt_logics.h"
#include "expr2smt.h"

std::string smt_typecheckt::prefix = "smt::";
std::string smt_typecheckt::tbase = prefix+"theory::";
std::string smt_typecheckt::lbase = prefix+"logic::";
std::string smt_typecheckt::bsymn = prefix+"benchmarks";

/*******************************************************************\

Function: check_subtype

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

bool check_subtype(const typet &t, const irep_idt &s)
{
  forall_subtypes(it, t)
    if(it->id()==s)
      return true;

  return false;
}

/*******************************************************************\

Function: smt_typecheckt::typecheck

  Inputs: none

 Outputs: nothing

 Purpose: typechecks a given parsetree and inserts the according 
          symbols into the context.

\*******************************************************************/

void smt_typecheckt::typecheck()
{
  typecheck_theories();
  typecheck_logics();
  
  {
    symbolt bsym;
    bsym.name=bsymn;
    bsym.value=exprt("benchmarks");

    if(context.symbols.find(bsymn)==
       context.symbols.end())
      context.add(bsym);
  }
  
  typecheck_benchmarks();
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_theories

  Inputs: none

 Outputs: nothing

 Purpose: typechecks the theories in a given parsetree and 
          inserts the according symbols into the context.

\*******************************************************************/

void smt_typecheckt::typecheck_theories()
{
  for(smt_parse_treet::theoriest::iterator
      it=parse_tree.theories.begin();
      it!=parse_tree.theories.end();
      it++)
  {
    smt_parse_treet::theoryt &t = *it;
    
    symbolt s = new_symbol(tbase + ii2string(t.name));
    s.base_name = ii2string(t.name);    
    s.pretty_name = s.base_name;
    s.module = this->module;
    s.location = t.location;
    s.type = typet("theory");
    s.mode = "SMT";
    s.is_type = true;
    
    typecheck_theory_basics(t);
    typecheck_theory_sorts(t, s);
    typecheck_theory_functions(t, s);
    typecheck_theory_predicates(t, s);
    typecheck_theory_axioms(t, s);
    
    symbolst::iterator it = context.symbols.find(s.name);
    if (it==context.symbols.end())
      context.add(s);
    else if (it->second.is_extern)
      it->second = s;
    else
    {
      err_location(s.location);
      str << "Duplicate symbol: '" << s.name << "'";
      throw 0;
    }
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_theory_basics

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_theory_basics(
  const smt_parse_treet::theoryt &t) 
{
  // basic checks (definition 3 of the SMT-LIB Standard v1.2)
  // required attributes
  if (t.sorts.size()==0)
  {
    err_location(t.location);
    str << "Missing sorts attribute for theory " << ii2string(t.name) << "'";
    throw 0;
  }
  else if(t.definitions.size()==0)
  {
    err_location(t.location);
    str << "Missing definition attribute for theory " << ii2string(t.name) << "'";
    throw 0;
  }
  
  // optional attributes
  if (t.sorts.size()>1)
  {
    err_location(t.location);
    str << "Multiple sorts attributes for theory " << ii2string(t.name) << "'";
    throw 0;
  }
  else if (t.definitions.size()>1)
  {
    err_location(t.location);
    str << "Multiple definition attributes for theory " << ii2string(t.name) << "'";
    throw 0;
  }
  else if (t.funs.size()>1)
  {
    err_location(t.location);
    str << "Multiple funs attributes for theory " << ii2string(t.name) << "'";
    throw 0;
  }
  else if (t.preds.size()>1)
  {
    err_location(t.location);
    str << "Multiple preds attributes for theory " << ii2string(t.name) << "'";
    throw 0;
  }
  else if (t.axioms.size()>1)
  {
    err_location(t.location);
    str << "Multiple axioms attributes for theory " << ii2string(t.name) << "'";
    throw 0;
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_theory_sorts

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_theory_sorts(
  const smt_parse_treet::theoryt &t,
  symbolt &s) 
{
  typet &sorts = s.type.add_type("sorts");
  
  for(smt_parse_treet::theoryt::sortst::const_iterator
      sit = t.sorts.begin();
      sit!=t.sorts.end();
      sit++)
  {
    forall_operands(oit, *sit)
      sorts.copy_to_subtypes(oit->type());
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_theory_functions

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_theory_functions(
  const smt_parse_treet::theoryt &t,
  symbolt &s) 
{
  const typet &sorts =
    static_cast<const typet&>(s.type.find("sorts"));
  
  typet &funs = s.type.add_type("functions");
  
  for(smt_parse_treet::theoryt::funst::const_iterator 
      fit = t.funs.begin();
      fit!= t.funs.end();
      fit++)
  {
    for(std::list<exprt>::const_iterator
      lit = fit->second.begin();
      lit!= fit->second.end();
      lit++)
    {
      forall_operands(oit, *lit)
      {
        typet fs;
  
        if (oit->id()=="natural" || oit->id()=="rational") 
          fs.id(oit->get("value"));
        else 
          fs.id(oit->id());

        irept::named_subt::const_iterator ni = 
          oit->get_named_sub().find("annotations");          
        if (ni!=oit->get_named_sub().end()) 
        {
          irept& ann = fs.add("annotations");
          forall_operands(ooit, static_cast<const exprt&>(ni->second))
          {
            ann.set(ooit->find("name").id(), "1");
          }
        }
                  
        if (oit->operands().size()>0)
          forall_operands(ooit, oit->op0()) {
            if (!check_subtype(sorts, ooit->type().id()))
            {
              err_location(ooit->location());
              str << "Unknown sort symbol '" << ooit->type().id_string() << "'";
              throw 0;
            }
            fs.copy_to_subtypes(ooit->type());
          }
  
        // check Definition 3, item 3 (unique return values)

        for(std::list<exprt>::const_iterator
          llit = fit->second.begin();
          llit!= fit->second.end();
          llit++)
        {
          if ( llit!=lit &&
              signature_eq_except_last((typet&)*llit, fs) &&
              ((typet&)*llit).subtypes().back().id() == 
              fs.subtypes().back().id() )
          {
            err_location(fs.location());
            str << "Function defined twice with different return types: '" 
              << fs.id_string()+"'";
            throw 0;
          }
        }
        
        funs.copy_to_subtypes(fs);
      }
    }
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_theory_predicates

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_theory_predicates(
  const smt_parse_treet::theoryt &t,
  symbolt &s) 
{
  const typet &sorts =
    static_cast<const typet&> (s.type.find("sorts"));
  
  typet &preds = s.type.add_type("predicates");
  
  for(smt_parse_treet::theoryt::predst::const_iterator
      pit = t.preds.begin();
      pit!=t.preds.end();
      pit++)
  {
    for(std::list<exprt>::const_iterator
      lit = pit->second.begin();
      lit!= pit->second.end();
      lit++)
    {
      forall_operands (oit, *lit)
      {
        typet ps(oit->id());
  
        if(oit->operands().size()>0)
          forall_operands(ooit, oit->op0())
          {
            if (!check_subtype(sorts, ooit->type().id()))
            {
              err_location(ooit->location());
              str << "Unknown sort symbol '" << ooit->type().id_string() << "'";
              throw 0;
            }
            ps.copy_to_subtypes(ooit->type());
          }
          
        for(std::list<exprt>::const_iterator
          llit = pit->second.begin();
          llit!= pit->second.end();
          llit++)
        {
          if ( llit!=lit && signature_eq((typet&)*llit, ps)) 
          {
            err_location(ps.location());
            warning("Predicate symbol defined twice with same signature: '" +
                  ps.id_string()+"'.");
          }
        }
      
        preds.copy_to_subtypes(ps);
      }
    }
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_theory_axioms

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_theory_axioms(
  const smt_parse_treet::theoryt &t,
  symbolt &s) 
{
  unsigned axiomcnt=0;
  
  typet &axioms = s.type.add_type("axioms");
  
  for(smt_parse_treet::theoryt::axiomst::const_iterator
      ait=t.axioms.begin();
      ait!=t.axioms.end();
      ait++)
  {
    symbolst ts; 
    ts.insert(std::pair<std::string,symbolt>("", s));
    message_streamt err(message_handler);
    
    smt_typecheck_expr_generict checker(err, exprt(), ts);
    
    forall_operands(oit, *ait)
    {
      typet as("axiom"+i2string(axiomcnt));
      axiomcnt++;
      
      checker.typecheck_expr(*oit);
      
      as.set("formula",*oit);
      axioms.copy_to_subtypes(as);
    }
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_logics

  Inputs: none

 Outputs: nothing

 Purpose: typechecks the logics in a given parsetree and 
          inserts the according symbols into the context.

\*******************************************************************/

void smt_typecheckt::typecheck_logics()
{  
  for(smt_parse_treet::logicst::const_iterator
      it=parse_tree.logics.begin();
      it!=parse_tree.logics.end();
      it++)
  {
    const smt_parse_treet::logict &l = *it;
    
    symbolt s = new_symbol(lbase + ii2string(l.name));
    s.base_name = ii2string(l.name);
    s.pretty_name = s.base_name;
    s.location = l.location;
    s.is_type = true;
    s.type = typet("logic");
    
    // basic checks (definition 4 of the SMT-LIB Standard v1.2)
    if (l.theories.size()==0)
    {
      err_location(l.location);
      str << "Missing theory attribute for logic '" << ii2string(l.name) << "'";
      throw 0;
    }
    else if (l.theories.size()!=1)
    {
      err_location(l.location);
      str << "Multiple theory attributes for logic '" << ii2string(l.name) << "'";
      throw 0;
    }

    if(l.languages.size()==0)
    {
      err_location(l.location);
      str << "Missing language attribute for logic '" << ii2string(l.name) << "'";
      throw 0;
    }
    else if (l.languages.size()!=1)
    {
      err_location(l.location);
      str << "Multiple language attributes for logic '" << ii2string(l.name) << "'";
      throw 0;
    }
    
    typet &theories = s.type.add_type("theories");
    

    for(smt_parse_treet::logict::theoriest::const_iterator ltit = 
        l.theories.begin();
        ltit!=l.theories.end();
        ltit++)
    {
      if(context.symbols.find(tbase+ii2string(*ltit))==
         context.symbols.end())
      {
        symbolt s = new_symbol(tbase+ii2string(*ltit));
        s.is_extern=true;
        s.is_type=true;
        context.add(s);
      }

      theories.copy_to_subtypes(typet(ii2string(*ltit)));
    }
    
    // Everything else in the logic is a natural language description.
    
    symbolst::iterator it = context.symbols.find(s.name);
    if (it==context.symbols.end())
      context.add(s);
    else if (it->second.is_extern)
      it->second = s;
    else
    {
      err_location(s.location);
      str << "Duplicate symbol: '" << s.name << "'";
      throw 0;
    }
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_benchmarks

  Inputs: none

 Outputs: nothing

 Purpose: typechecks the logics in a given parsetree and 
          inserts the according symbols into the context.

\*******************************************************************/

void smt_typecheckt::typecheck_benchmarks()
{  
  for(smt_parse_treet::benchmarkst::iterator
      it=parse_tree.benchmarks.begin();
      it!=parse_tree.benchmarks.end();
      it++)
  {
    smt_parse_treet::benchmarkt &b = *it;
    
    symbolt &bsym = context.symbols.find(bsymn)->second;
    
    irept bm("benchmark");    
    
    bm.set("name", ii2string(b.name));
    bm.set("#location", b.location);
    
    typecheck_benchmark_basics(b);
    typecheck_benchmark_logics(b, bm);

    // check status
    bm.set("status",
      (b.status.front()==smt_parse_treet::benchmarkt::SAT)?
      "sat":
      (b.status.front()==smt_parse_treet::benchmarkt::UNSAT)?
      "unsat":"unknown");
                          
    typecheck_benchmark_sorts(b, bm);
    typecheck_benchmark_functions(b, bm);
    typecheck_benchmark_predicates(b, bm);
    typecheck_benchmark_assumptions(b, bm);
    typecheck_benchmark_formulas(b, bm);

    bsym.value.move_to_operands((exprt&)bm);      
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_benchmark_basics

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_benchmark_basics(
      const smt_parse_treet::benchmarkt& b) 
{
  // basic checks (definition 5 of the SMT-LIB Standard v1.2)
  if (b.logics.size()==0) 
  {
    err_location(b.location);
    str << "Missing logic attribute for benchmark '" << b.name << "'";
    throw 0;
  }
  else if (b.logics.size()!=1) 
  {
    err_location(b.location);
    str << "Multiple logic attributes for benchmark '" << b.name << "'";
    throw 0;
  }
  
  if (b.formulas.size()==0)
  {
    err_location(b.location);
    str << "Missing formula attribute for benchmark '" << b.name << "'";
    throw 0;
  }
  else if (b.formulas.size()!=1) 
  {
    err_location(b.location);
    str << "Multiple formula attributes for benchmark '" << b.name << "'";
    throw 0;
  }
  
  if (b.status.size()==0)
  {
    err_location(b.location);
    str << "Missing status attribute for benchmark '" << b.name << "'";
    throw 0;
  }
  else if (b.status.size()!=1) 
  {
    err_location(b.location);
    str << "Multiple status attributes for benchmark '" << b.name << "'";
    throw 0;
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_benchmark_logics

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_benchmark_logics(
  const smt_parse_treet::benchmarkt& b,
  irept &bm) 
{
  bm.set("logic", irept(b.logics.front()));
  
  if(context.symbols.find(lbase+ii2string(b.logics.front()))==
     context.symbols.end())
  {
    symbolt logicsym;
    logicsym=new_symbol(lbase+ii2string(b.logics.front()));
    logicsym.is_extern=true;
    logicsym.is_type=true;
    context.add(logicsym);
  } 
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_benchmark_sorts

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_benchmark_sorts(
  const smt_parse_treet::benchmarkt& b,
  irept &bm) 
{
  typet &sorts = static_cast<typet&>(bm.add("sorts"));
      
  for(smt_parse_treet::benchmarkt::sort_symbolst::const_iterator
      ssit=b.sort_symbols.begin();
      ssit!=b.sort_symbols.end();
      ssit++)
  {
    // check duplicate sorts, as of Definition 5, item 4.

    Forall_subtypes(stit, sorts)
      if (stit->id()==ssit->find("type").id())
      {
        err_location(ssit->location());
        str << "Duplicate extrasort symbol '" << ssit->find("type").id_string() 
          << "' in benchmark.";
        throw 0;
      }

    typet ss = static_cast<const typet&>(ssit->find("type"));
    ss.location() = ssit->location();
    assert(ss.is_not_nil());
    sorts.copy_to_subtypes(ss);
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_benchmark_functions

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_benchmark_functions(
  const smt_parse_treet::benchmarkt &b,
  irept &bm) 
{
  typet &funs = static_cast<typet&>(bm.add("functions"));
      
  for(smt_parse_treet::benchmarkt::function_symbolst::const_iterator
      fsit = b.function_symbols.begin();
      fsit!=b.function_symbols.end();
      fsit++)
  {
    for(std::list<exprt>::const_iterator
      lit = fsit->second.begin();
      lit!= fsit->second.end();
      lit++)
    {
      typet fs;
  
      if (lit->id()=="natural" || lit->id()=="rational") 
        fs.id(lit->get("value"));
      else 
        fs.id(lit->id());
  
      fs.location() = lit->location();
      
      irept::named_subt::const_iterator ni = 
        lit->get_named_sub().find("annotations");          
      if (ni!=lit->get_named_sub().end()) 
      {
        irept& ann = fs.add("annotations");
        forall_operands(ooit, static_cast<const exprt&>(ni->second))
        {
          ann.set(ooit->find("name").id(), "1");
        }
      }
  
      if (lit->operands().size()>0)
        forall_operands(ooit, lit->op0())
        {
          // dont check sorts here, will be done after linking
          typet s = static_cast<const typet&>(ooit->find("type"));
          s.location() = ooit->location();
          fs.copy_to_subtypes(s);
        }
  
      // check Definition 5, item 5 (unique return values)
      for(std::list<exprt>::const_iterator
        llit = fsit->second.begin();
        llit!= fsit->second.end();
        llit++)
      {
        if ( llit!=lit &&
             signature_eq_except_last((typet&)*llit, fs) && 
             ((typet&)*llit).subtypes().back().id() != 
             fs.subtypes().back().id() ) 
        {          
          err_location(fs.location());
          str << "Function defined twice with different return types: '" 
            << fs.id_string() << "'";
          throw 0;
        }
      }
  
      funs.copy_to_subtypes(fs);
    }
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_benchmark_predicates

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_benchmark_predicates(
  const smt_parse_treet::benchmarkt &b,
  irept &bm) 
{
  typet &preds = static_cast<typet&>(bm.add("predicates"));
    
  for(smt_parse_treet::benchmarkt::predicate_symbolst::const_iterator
      psit = b.predicate_symbols.begin();
      psit!= b.predicate_symbols.end();
      psit++)
  {
    for(std::list<exprt>::const_iterator
      lit = psit->second.begin();
      lit!= psit->second.end();
      lit++)
    {
      typet ps(lit->id());
      ps.location() = lit->location();
      
      if (lit->operands().size()>0)
        forall_operands(ooit, lit->op0())
        {
          // don't check sorts here, will be done after linking       
          typet s = static_cast<const typet&>(ooit->find("type"));
          s.location() = ooit->location();
          ps.copy_to_subtypes(s);
        }
  
      //check for doubles
      for(std::list<exprt>::const_iterator
        llit = psit->second.begin();
        llit!= psit->second.end();
        llit++)
      {
        if ( llit!=lit && signature_eq((typet&)*llit, ps)) 
        {
          // the standard does not really forbid this, so just warn.
          err_location(ps.location());
          warning("Predicate symbol defined twice with same signature: '" +
                  ii2string(ps)+"'.");
        }
      }
              
      preds.copy_to_subtypes(ps);
    }
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_benchmark_assumptions

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_benchmark_assumptions(
  const smt_parse_treet::benchmarkt& b,
  irept &bm) 
{
  exprt &assumptions = static_cast<exprt&>(bm.add("assumptions"));

  for(smt_parse_treet::benchmarkt::assumptionst::const_iterator
      ait = b.assumptions.begin();
      ait!=b.assumptions.end();
      ait++)
  {
    // formula will be checked upon linking, when the theory is available!
    assumptions.copy_to_operands(*ait);
  }
}

/*******************************************************************\

Function: smt_typecheckt::typecheck_benchmark_formulas

  Inputs: 

 Outputs: 

 Purpose: 

\*******************************************************************/

void smt_typecheckt::typecheck_benchmark_formulas(
  const smt_parse_treet::benchmarkt &b,
  irept &bm) 
{
  // check formulas
  exprt &formulas = static_cast<exprt&>(bm.add("formulas"));

  for(smt_parse_treet::benchmarkt::formulast::const_iterator
      fit = b.formulas.begin();
      fit!=b.formulas.end();
      fit++)
  {
    // formula will be checked upon linking, when the theory is available!
    formulas.copy_to_operands(*fit);
  }
}

/*******************************************************************\

Function: smt_typecheckt::new_symbol

  Inputs: none

 Outputs: symbolt 

 Purpose: creates a new symbolt that is prefilled with some values

\*******************************************************************/

symbolt smt_typecheckt::new_symbol(const std::string &name)
{
  symbolt sym;
  sym.name = name;
  sym.module = this->module;
  sym.mode="SMT";
  sym.type.make_nil();
  return sym;
}

/*******************************************************************\

Function: smt_typecheckt::signature_eq

  Inputs: two exprt lists

 Outputs: true if the sorts in the list are equal, false otherwise

 Purpose: checks if two function signatures match.

\*******************************************************************/

bool smt_typecheckt::signature_eq(
    const typet& f1, 
    const typet& f2 )
{
  bool equal=true;
  if (f1.subtypes().size()==f2.subtypes().size())
  {
    for(unsigned i=0; i<f1.subtypes().size(); i++)
    {
      if (f1.subtypes()[i].id()!=f2.subtypes()[i].id())
      {
        equal=false; 
        break;
      }
    }
  }
  else
    equal=false;
    
  return equal;
}

/*******************************************************************\

Function: smt_typecheckt::signature_eq

  Inputs: two exprt lists

 Outputs: true if the sorts in the list are equal, false otherwise

 Purpose: checks if two function signatures match, ignoring the last 
          one (which is usually the return value of the function)

\*******************************************************************/

bool smt_typecheckt::signature_eq_except_last( 
    const typet& f1, 
    const typet& f2 )
{
  bool equal=true;
  if (f1.subtypes().size()==f2.subtypes().size())
  {
    for(unsigned i=0; i<f1.subtypes().size()-1; i++)
    {
      if (f1.subtypes()[i].id()!=f2.subtypes()[i].id())
      {
        equal=false; 
        break;
      }
    }
  }
  else
    equal=false;
    
  return equal;
}
