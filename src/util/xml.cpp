/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cstdlib>
#include <util/i2string.h>
#include <util/xml.h>

void xmlt::clear()
{
  data.clear();
  name.clear();
  attributes.clear();
  elements.clear();
}

void xmlt::swap(xmlt &xml)
{
  xml.data.swap(data);
  xml.attributes.swap(attributes);
  xml.elements.swap(elements);
  xml.name.swap(name);
}

void xmlt::output(std::ostream &out, unsigned indent) const
{
  do_indent(out, indent);

  out << '<' << name;

  for(const auto &attribute : attributes)
    out << ' ' << attribute.first << '=' << '"' << attribute.second << '"';

  out << '>';

  if(elements.empty())
    out << data;
  else
  {
    out << std::endl;

    for(const auto &element : elements)
      element.output(out, indent + 2);

    do_indent(out, indent);
  }

  out << '<' << '/' << name << '>' << std::endl;
}

std::string xmlt::escape(const std::string &s)
{
  std::string result;

  for(char ch : s)
  {
    switch(ch)
    {
    case '&':
      result += "&amp;";
      break;

    case '<':
      result += "&lt;";
      break;

    case '>':
      result += "&gt;";
      break;

    default:
      if(ch < ' ')
        result += "&#" + i2string((unsigned char)ch) + ";";
      else
        result += ch;
    }
  }

  return result;
}

std::string xmlt::escape_attribute(const std::string &s)
{
  std::string result;

  for(char ch : s)
  {
    switch(ch)
    {
    case '&':
      result += "&amp;";
      break;

    case '<':
      result += "&lt;";
      break;

    case '>':
      result += "&gt;";
      break;

    case '"':
      result += "&quot;";
      break;

    default:
      result += ch;
    }
  }

  return result;
}

void xmlt::do_indent(std::ostream &out, unsigned indent)
{
  for(unsigned i = 0; i < indent; i++)
    out << ' ';
}

xmlt::elementst::const_iterator xmlt::find(const std::string &name) const
{
  for(elementst::const_iterator it = elements.begin(); it != elements.end();
      it++)
    if(it->name == name)
      return it;

  return elements.end();
}

xmlt::elementst::iterator xmlt::find(const std::string &name)
{
  for(elementst::iterator it = elements.begin(); it != elements.end(); it++)
    if(it->name == name)
      return it;

  return elements.end();
}

void xmlt::set_attribute(const std::string &attribute, unsigned value)
{
  set_attribute(attribute, i2string(value));
}

void xmlt::set_attribute(const std::string &attribute, const std::string &value)
{
  if(
    (value[0] == '\"' && value[value.size() - 1] == '\"') ||
    (value[0] == '\'' && value[value.size() - 1] == '\''))
  {
    attributes[attribute] = value.substr(1, value.size() - 2);
  }
  else
  {
    attributes[attribute] = value;
  }
}

std::string xmlt::unescape(const std::string &str)
{
  std::string result;

  result.reserve(str.size());

  for(std::string::const_iterator it = str.begin(); it != str.end(); it++)
  {
    if(*it == '&')
    {
      std::string tmp;
      it++;

      while(it != str.end() && *it != ';')
        tmp += *it++;

      if(tmp == "gt")
        result += '>';
      else if(tmp == "lt")
        result += '<';
      else if(tmp == "amp")
        result += '&';
      else if(tmp[0] == '#' && tmp[1] != 'x')
      {
        char c = atoi(tmp.substr(1, tmp.size() - 1).c_str());
        result += c;
      }
      else
        throw "XML escape code not implemented";
    }
    else
      result += *it;
  }

  return result;
}
