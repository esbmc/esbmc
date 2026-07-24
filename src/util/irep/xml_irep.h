#ifndef XML_IREP_H
#define XML_IREP_H

#include <util/irep/irep.h>
#include <util/base/xml.h>
#include <util/message/message.h>

void convert(const irept &irep, xmlt &xml);

void convert(const xmlt &xml, irept &irep);

#endif
