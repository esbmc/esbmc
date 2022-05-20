#include "irep2/irep2_type.h"
#include <esbmc/bmc_domain_split.h>
#include <boost/algorithm/string.hpp>

void BMC_Domain_Split::bmc_get_free_variables()
{
  for(auto i : steps)
  {
    //check for an assignment in the source code
    if(i.is_assignment() && !i.hidden)
    {
      //check for supported types
      switch(i.lhs->type->type_id)
      {
      case type2t::signedbv_id:   //signed int
      case type2t::unsignedbv_id: //unsigned int
      case type2t::floatbv_id:    //unsigned int
      case type2t::fixedbv_id:    //unsigned int
        break;
      default:
        continue;
      }

      if(i.rhs->expr_id == expr2t::symbol_id) //need to use rhs not original_rhs
      {
        const symbol2t &rhs = to_symbol2t(i.rhs);
        std::string nd("nondet"); // find a more concrete method of doing this
        if(boost::algorithm::contains(rhs.thename.as_string(), nd))
        {
          free_vars.push_back(i.lhs);
        }
      }
      //check if symbol is wrapped in a typecast
      else if(i.rhs->expr_id == expr2t::typecast_id)
      {
        const typecast2t &rhs = to_typecast2t(i.rhs);
        if(rhs.from->expr_id == expr2t::symbol_id)
        {
          std::string nd("nondet");
          if(boost::algorithm::contains(
               to_symbol2t(rhs.from).thename.as_string(), nd))
          {
            free_vars.push_back(i.lhs);
          }
        }
      }
    }
  }
}

//counts how many times v occurs in rhs
//this is perfomed by recursively searching through sub exprs
int occurances(const expr2tc &rhs, expr2tc &v)
{
  if(rhs->expr_id == expr2t::symbol_id)
  {
    if(
      to_symbol2t(rhs).thename.as_string() ==
      to_symbol2t(v).thename.as_string())
      return 1;
  }
  else
  {
    int se = rhs->get_num_sub_exprs();
    int occ = 0;
    for(int i = 0; i < se; i++)
    {
      occ += occurances(*(rhs->get_sub_expr(i)), v);
    }
    return occ;
  }
  return 0;
}

void BMC_Domain_Split::bmc_free_var_occurances()
{
  for(expr2tc v : free_vars)
  {
    int occ = 0;
    for(auto i : steps)
    {
      //assignment is the only relevant type when doing ssa
      if(i.is_assignment() && !i.hidden)
      {
        occ += occurances(i.rhs, v);
      }
    }
    free_var_counts.push_back(occ);
  }
}

int most_played_index(std::vector<int> free_var_counts)
{
  return std::max_element(free_var_counts.begin(), free_var_counts.end()) -
         free_var_counts.begin();
}

std::pair<expr2tc, int> BMC_Domain_Split::bmc_most_used() const
{
  std::pair<expr2tc, int> pair;

  int i = most_played_index(free_var_counts);

  pair.second = free_var_counts.at(i);
  pair.first = free_vars.at(i);

  return pair;
}

std::vector<expr2tc>
BMC_Domain_Split::bmc_split_domain_exprs(const expr2tc &var) const
{
  std::vector<expr2tc> exprs;
  expr2tc c;

  switch(var->type->type_id)
  {
  case type2t::fixedbv_id: //fixed point representation (--fixedbv)
  {
    fixedbvt a;
    a.spec.integer_bits = to_fixedbv_type(var->type).integer_bits;
    a.spec.width = to_fixedbv_type(var->type).width;
    a.from_integer(0); //convert 0 to bitvector
    c = constant_fixedbv2tc(a);
    break;
  }
  case type2t::floatbv_id: //floating point representation (--floatbv)
  {
    ieee_floatt a;
    a.spec.e = to_floatbv_type(var->type).exponent;
    a.spec.f = to_floatbv_type(var->type).fraction;
    a.make_zero();
    c = constant_floatbv2tc(a);
    break;
  }
  case type2t::signedbv_id: //signed int
  {
    c = constant_int2tc(var->type, 0);
    break;
  }
  case type2t::unsignedbv_id: //unsigned int
  {
    c = constant_int2tc(var->type, 2000);
    break;
  }
  default:
  {
    //not supported
    return exprs;
  }
  }

  /* var->dump(); */

  //Fixed point bitvector double
  /* fixedbvt a; */
  /* a.spec.integer_bits = 32; */
  /* a.spec.width = 64; */
  /* a.from_integer(0); //convert 0 to bitvector */
  /* expr2tc c = constant_fixedbv2tc(a); */

  //Fixed point bitvector float
  /* fixedbvt a; */
  /* a.spec.integer_bits = 16; */
  /* a.spec.width = 32; */
  /* a.from_integer(0); //convert 0 to bitvector */
  /* expr2tc c = constant_fixedbv2tc(a); */

  //Floating point float
  /* ieee_floatt a; */
  /* a.spec.e = 8; */
  /* a.spec.f = 23; */
  /* a.make_zero(); */
  /* expr2tc c = constant_floatbv2tc(a); */

  /* //Floating point double */
  /* ieee_floatt a; */
  /* a.spec.e = 11; */
  /* a.spec.f = 52; */
  /* a.make_zero(); */
  /* expr2tc c = constant_floatbv2tc(a); */

  //short
  /* c = constant_int2tc(get_int16_type(), 0); */

  //int
  /* c = constant_int2tc(get_int32_type(), 0); */
  //long
  /* c = constant_int2tc(get_int64_type(), 0); */

  lessthan2tc condlt(var, c);
  greaterthanequal2tc condgt(var, c);
  expr2tc g = constant_bool2tc(true);

  exprs.push_back(condlt);
  exprs.push_back(condgt);

  return exprs;
}

std::vector<int> divide_interval(int l, int h, int d)
{
  std::vector<int> intervals;

  int interval_size = (h - l) / d;
  for(int i = 1; i < d; i++)
  {
    intervals.push_back(l + i * interval_size);
  }

  return intervals;
}

//split the domain a given number of times
std::vector<expr2tc> BMC_Domain_Split::bmc_split_domain_exprs(
  const expr2tc &var,
  unsigned int depth) const
{
  //minimum number of partitions is two
  if(depth < 2)
  {
    depth = 2;
  }

  std::vector<expr2tc> exprs;

  std::vector<int> intervals;

  //vector of points to partition the domain on
  std::vector<expr2tc> cs;

  switch(var->type->type_id)
  {
  case type2t::fixedbv_id: //fixed point representation (--fixedbv)
  {
    intervals = divide_interval(-100000, 100000, depth);

    for(int i : intervals)
    {
      fixedbvt a;
      a.spec.integer_bits = to_fixedbv_type(var->type).integer_bits;
      a.spec.width = to_fixedbv_type(var->type).width;
      a.from_integer(
        i); //convert integer to fixed point bitvector representation
      cs.push_back(constant_fixedbv2tc(a));
    }
    break;
  }
  case type2t::floatbv_id: //floating point representation (--floatbv)
  {
    intervals = divide_interval(-100000, 100000, depth);

    for(int i : intervals)
    {
      ieee_floatt a;
      a.spec.e = to_floatbv_type(var->type).exponent;
      a.spec.f = to_floatbv_type(var->type).fraction;
      a.from_integer(i); //convert i to floating point bitvector
      cs.push_back(constant_floatbv2tc(a));
    }
    break;
  }
  case type2t::signedbv_id: //signed int
  {
    //create intervals
    intervals = divide_interval(-100000, 100000, depth);

    for(int i : intervals)
    {
      cs.push_back(constant_int2tc(var->type, i));
    }

    break;
  }
  case type2t::unsignedbv_id: //unsigned int
  {
    //create intervals
    intervals = divide_interval(0, 100000, depth);

    for(int i : intervals)
    {
      cs.push_back(constant_int2tc(var->type, i));
    }

    break;
  }
  default:
  {
    //not supported
    msg.status("Variable type not supported");
    return exprs;
  }
  }

  lessthan2tc condlt(var, cs.front());
  greaterthanequal2tc condgte(var, cs.back());

  //add the intervals outside the boundary
  exprs.push_back(condlt);
  exprs.push_back(condgte);

  /* msg.status(fmt::format("{}", cs.size())); */
  //between points
  for(unsigned int i = 0; i < cs.size() - 1; i++)
  {
    //create and expression which expresses the interval
    lessthan2tc c2(var, cs.at(i + 1));
    greaterthanequal2tc c1(var, cs.at(i));
    and2tc a(c1, c2);

    exprs.push_back(a);
  }

  //print out the domain partition intervals that have been created
  std::stringstream ss;

  ss << "\t[MIN, " << intervals.front() << ")\n";

  for(unsigned int i = 0; i < intervals.size() - 1; i++)
  {
    ss << "\t[" << intervals.at(i) << ", " << intervals.at(i + 1) << ")\n";
  }

  ss << "\t[" << intervals.back() << ", MAX]";

  msg.status("Partitioned into intervals: ");
  msg.status(ss.str());

  return exprs;
}

void BMC_Domain_Split::print_free_vars() const
{
  std::stringstream ss;
  ss << "Free variables found: ";
  for(auto v : free_vars)
  {
    symbol2t s = to_symbol2t(v);

    std::vector<std::string> sv;
    boost::split(sv, s.thename.as_string(), boost::is_any_of("@"));

    ss << sv.back() << " ";
  }
  msg.status(ss.str());
}
