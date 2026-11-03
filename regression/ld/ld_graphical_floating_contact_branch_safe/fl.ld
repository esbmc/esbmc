<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="x" productName="x" productVersion="1" creationDateTime="2026-10-11T00:00:00"/>
  <contentHeader name="fl"><coordinateInfo><ld><scaling x="10" y="10"/></ld></coordinateInfo></contentHeader>
  <types><dataTypes/><pous><pou name="fl" pouType="program">
    <interface>
      <inputVars><variable name="a"><type><BOOL/></type></variable><variable name="b"><type><BOOL/></type></variable></inputVars>
      <outputVars><variable name="y"><type><BOOL/></type></variable></outputVars>
    </interface>
    <body><LD>
      <leftPowerRail localId="0"><position x="0" y="0"/><connectionPointOut formalParameter="none"/></leftPowerRail>
      <contact localId="3" negated="false" storage="none" edge="none"><position x="20" y="10"/><connectionPointIn><connection refLocalId="0"/></connectionPointIn><connectionPointOut/><variable>a</variable></contact>
      <contact localId="7" negated="false" storage="none" edge="none"><position x="20" y="60"/><connectionPointIn/><connectionPointOut/><variable>b</variable></contact>
      <coil localId="8" negated="false" storage="none"><position x="160" y="10"/><connectionPointIn><connection refLocalId="3"/><connection refLocalId="7"/></connectionPointIn><connectionPointOut/><variable>y</variable></coil>
      <rightPowerRail localId="2147483646"><position x="400" y="0"/><connectionPointIn><connection refLocalId="8"/></connectionPointIn></rightPowerRail>
    </LD></body>
  </pou></pous></types>
  <instances><configurations><configuration name="c"><resource name="r"><task name="t" interval="T#10ms" priority="0"><pouInstance name="p" typeName="fl"/></task></resource></configuration></configurations></instances>
</project>
