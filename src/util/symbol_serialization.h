#ifndef SYMBOL_SERIALIZATION_H_
#define SYMBOL_SERIALIZATION_H_

#include <util/irep_serialization.h>
#include <util/symbol.h>

class symbol_serializationt
{
private:
  irep_serializationt irepconverter;
  std::list<irept> irepcache;

public:
  symbol_serializationt(irep_serializationt::ireps_containert &ic)
    : irepconverter(ic){};

  void convert(const symbolt &, std::ostream &);
  void convert(std::istream &, irept &);
};

#endif /*SYMBOL_SERIALIZATION_H_*/
