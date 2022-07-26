#include <util/xml_irep.h>

void convert(const irept &irep, xmlt &xml)
{
  if(irep.id() != "nil")
    xml.new_element("id").data = xmlt::escape(irep.id_string());

  forall_irep(it, irep.get_sub())
  {
    xmlt &x_sub = xml.new_element("sub");
    convert(*it, x_sub);
  }

  forall_named_irep(it, irep.get_named_sub())
  {
    xmlt &x_nsub = xml.new_element("named_sub");
    x_nsub.set_attribute("name", name2string(it->first));
    convert(it->second, x_nsub);
  }

  forall_named_irep(it, irep.get_comments())
  {
    xmlt &x_com = xml.new_element("comment");
    x_com.set_attribute("name", name2string(it->first));
    convert(it->second, x_com);
  }
}

void convert(const xmlt &xml, irept &irep)
{
  irep.id("nil");
  xmlt::elementst::const_iterator it = xml.elements.begin();
  for(; it != xml.elements.end(); it++)
  {
    if(it->name == "id")
    {
      irep.id(xmlt::unescape(it->data));
    }
    else if(it->name == "named_sub")
    {
      irept r;
      convert(*it, r);
      std::string named_name = it->get_attribute("name");
      irep.move_to_named_sub(named_name, r);
    }
    else if(it->name == "sub")
    {
      irept r;
      convert(*it, r);
      irep.move_to_sub(r);
    }
    else if(it->name == "comment")
    {
      irept r;
      convert(*it, r);
      std::string named_name = it->get_attribute("name");
      irep.move_to_named_sub(named_name, r);
    }
    else
      // Should not happen
      log_error("Unknown sub found ({}); malformed xml?", it->name);
  }
}
