#ifndef XML_IREP_H
#define XML_IREP_H

#include <util/irep.h>
#include <util/xml.h>
#include <util/message.h>

void convert(const irept &irep, xmlt &xml);

void convert(const xmlt &xml, irept &irep);

#endif
