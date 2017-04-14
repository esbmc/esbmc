/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <util/cmdline.h>

std::string verification_file;

cmdlinet::cmdlinet()
{
}

cmdlinet::~cmdlinet()
{
  clear();
}

void cmdlinet::clear()
{
  options.clear();
  args.clear();
}

bool cmdlinet::isset(char option) const
{
  int i;

  i=getoptnr(option);
  if(i<0) return false;
  return options[i].isset;
}

bool cmdlinet::isset(const char *option) const
{
  int i;

  i=getoptnr(option);
  if(i<0) return false;
  return options[i].isset;
}

const char *cmdlinet::getval(char option) const
{
  int i;

  i=getoptnr(option);
  if(i<0) return (const char *)NULL;
  if(options[i].values.empty()) return (const char *)NULL;
  return options[i].values.front().c_str();
}

const std::list<std::string> &cmdlinet::get_values(char option) const
{
  int i;

  i=getoptnr(option);
  assert(i>=0);
  return options[i].values;
}

const char *cmdlinet::getval(const char *option) const
{
  int i;

  i=getoptnr(option);
  if(i<0) return (const char *)NULL;
  if(options[i].values.empty()) return (const char *)NULL;
  return options[i].values.front().c_str();
}

const std::list<std::string>&cmdlinet::get_values(const char *option) const
{
  int i;

  i=getoptnr(option);
  assert(i>=0);
  return options[i].values;
}

int cmdlinet::getoptnr(char option) const
{
  for(unsigned i=0; i<options.size(); i++)
    if(options[i].optchar==option)
      return i;
  
  return -1;
}

int cmdlinet::getoptnr(const char *option) const
{
  for(unsigned i=0; i<options.size(); i++)
    if(options[i].optstring==option)
      return i;
  
  return -1;
}

bool cmdlinet::parse(int argc, const char **argv, const struct opt_templ *opts)
{
  unsigned int i;

  clear();

  for (i = 0; opts[i].optchar != 0 || opts[i].optstring != ""; i++)
  {
    optiont option;

    option.optchar = opts[i].optchar;
    option.optstring = opts[i].optstring;

    if (option.optchar == 0)
      option.islong = true;
    else
      option.islong = false;

    option.isset = false;

    if (opts[i].type != switc)
      option.hasval = true;
    else
      option.hasval = false;

    options.push_back(option);
  }

  for(int i=1; i<argc; i++)
  {
    if(argv[i][0]!='-')
    {
      args.push_back(argv[i]);
      verification_file = argv[i];
    }
    else
    {
      int optnr;

      if(argv[i][1]=='-')
        optnr=getoptnr(argv[i]+2);
      else
        optnr=getoptnr(argv[i][1]);
   
      if(optnr<0)
      {
        failing_option = std::string(argv[i]);
        return true;
      }

      options[optnr].isset=true;
      if(options[optnr].hasval)
      {
        if(argv[i][2]==0 || options[optnr].islong)
        {
          i++;
          if(i==argc) return true;
          if(argv[i][0]=='-') return true;
          options[optnr].values.push_back(argv[i]);
        }
        else
          options[optnr].values.push_back(argv[i]+2);
      }
    }
  }

  for (i = 0; opts[i].optchar != 0 || opts[i].optstring != ""; i++)
  {
    int optnr;

    if (opts[i].init == "")
      continue;

    if (opts[i].optchar != 0)
      optnr = getoptnr(opts[i].optchar);
    else
      optnr = getoptnr(opts[i].optstring.c_str());

    if (options[optnr].values.size() != 0)
      continue;

    options[optnr].values.push_back(opts[i].init);
  }

  return false;
}
