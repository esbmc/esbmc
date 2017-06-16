/*******************************************************************\

Module: Subgoal Documentation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <clang-c-frontend/expr2c.h>
#include <esbmc/document_subgoals.h>
#include <fstream>
#include <util/i2string.h>

#define MAXWIDTH 62

struct linet
{
  std::string text;
  int line_number;
};

/*******************************************************************\

Function: strip_space

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void strip_space(std::list<linet> &lines)
{
  unsigned strip=50;

  for(std::list<linet>::const_iterator it=lines.begin();
      it!=lines.end(); it++)
  {
    for(unsigned j=0; j<strip && j<it->text.size(); j++)
      if(it->text[j]!=' ')
      {
        strip=j;
        break;
      }
  }

  if(strip!=0)
  {
    for(std::list<linet>::iterator it=lines.begin();
        it!=lines.end(); it++)
    {
      if(it->text.size()>=strip)
        it->text=std::string(it->text, strip, std::string::npos);

      if(it->text.size()>=MAXWIDTH)
        it->text=std::string(it->text, 0, MAXWIDTH);
    }
  }
}

/*******************************************************************\

Function: escape_latex

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string escape_latex(const std::string &s, bool alltt)
{
  std::string dest;

  for(unsigned i=0; i<s.size(); i++)
  {
    if(s[i]=='\\' || s[i]=='{' || s[i]=='}')
      dest+="\\";

    if(!alltt &&
       (s[i]=='_' || s[i]=='$' || s[i]=='~' ||
        s[i]=='^' || s[i]=='%' || s[i]=='#' ||
        s[i]=='&'))
      dest+="\\";

    dest+=s[i];
  }

  return dest;
}

/*******************************************************************\

Function: emphasize

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string emphasize(const std::string &s)
{
  #if 0
  std::string dest;
  bool bold_mode=false;

  for(unsigned i=0; i<s.size(); i++)
  {
    bool new_mode=isalnum(s[i]) ||
                  s[i]=='.' || s[i]==',';

    if(new_mode!=bold_mode)
    {
      if(new_mode)
        dest+="{\\bf{";
      else
        dest+="}}";

      bold_mode=new_mode;
    }

    dest+=s[i];
  }

  if(bold_mode)
    dest+="}}";

  return dest;
  #else
  return "{\\ttb{}"+s+"}";
  #endif
}

/*******************************************************************\

Function: is_empty

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool is_empty_str(const std::string &s)
{
  for(unsigned i=0; i<s.size(); i++)
    if(isgraph(s[i]))
      return false;

  return true;
}

/*******************************************************************\

Function: get_code

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void get_code(const irept &location, std::string &dest)
{
  dest="";

  const irep_idt &file=location.file();
  const irep_idt &line=location.line();

  if(file=="" || line=="") return;

  std::ifstream in(file.c_str());

  if(!in) return;

  int line_int=atoi(line.c_str());

  int line_start=line_int-3,
      line_end=line_int+3;

  if(line_start<=1) line_start=1;

  // skip line_start-1 lines

  for(int l=0; l<line_start-1; l++)
  {
    std::string tmp;
    std::getline(in, tmp);
  }

  // read till line_end

  std::list<linet> lines;

  for(int l=line_start; l<=line_end && in; l++)
  {
    lines.push_back(linet());

    std::string &line=lines.back().text;
    std::getline(in, line);

    if(!line.empty() && line[line.size()-1]=='\r')
      line.resize(line.size()-1);

    lines.back().line_number=l;
  }

  // remove empty lines at the end and at the beginning

  for(std::list<linet>::iterator it=lines.begin();
      it!=lines.end();)
  {
    if(is_empty_str(it->text))
      it=lines.erase(it);
    else
      break;
  }

  for(std::list<linet>::iterator it=lines.end();
      it!=lines.begin();)
  {
    it--;

    if(is_empty_str(it->text))
      it=lines.erase(it);
    else
      break;
  }

  // strip space
  strip_space(lines);

  // build dest

  for(std::list<linet>::iterator it=lines.begin();
      it!=lines.end(); it++)
  {
    std::string line_no=i2string(it->line_number);

    while(line_no.size()<4)
      line_no=" "+line_no;

    std::string tmp=line_no+"  "+escape_latex(it->text, true);

    if(it->line_number==line_int)
      tmp=emphasize(tmp);

    dest+=tmp+"\n";
  }
}

/*******************************************************************\

Function: symex_bmct::document_subgoals

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

struct doc_claimt
{
  std::set<std::string> comment_set;
};

void document_subgoals(
  const symex_target_equationt &equation,
  std::ostream &out)
{
  typedef std::map<irept, doc_claimt> claim_sett;
  claim_sett claim_set;

  for(symex_target_equationt::SSA_stepst::const_iterator
      it=equation.SSA_steps.begin();
      it!=equation.SSA_steps.end();
      it++)
    if(it->is_assert())
    {
      locationt new_location;

      new_location.file(it->source.pc->location.file());
      new_location.line(it->source.pc->location.line());
      new_location.function(it->source.pc->location.function());

      claim_set[new_location].comment_set.insert(it->comment);
    }

  for(claim_sett::const_iterator it=claim_set.begin();
      it!=claim_set.end(); it++)
  {
    std::string code;
    const irept &location=it->first;

    get_code(location, code);

    out << "\\claimlocation{File "
        << escape_latex(location.file().as_string(), false)
        << " function "
        << escape_latex(location.function().as_string(), false)
        << "}" << std::endl;

    out << std::endl;

    for(std::set<std::string>::const_iterator
        s_it=it->second.comment_set.begin();
        s_it!=it->second.comment_set.end();
        s_it++)
      out << "\\claim{" << escape_latex(*s_it, false)
          << "}" << std::endl;

    out << std::endl;

    out << "\\begin{alltt}\\claimcode\n"
        << code
        << "\\end{alltt}\n";

    out << std::endl;
    out << std::endl;
  }
}
