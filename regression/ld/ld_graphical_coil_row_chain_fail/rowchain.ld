<?xml version="1.0" encoding="utf-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="ESBMC" productName="probe" productVersion="1.0" creationDateTime="2026-10-07T00:00:00" />
  <contentHeader name="rowchain"><coordinateInfo><fbd><scaling x="1" y="1" /></fbd><ld><scaling x="1" y="1" /></ld><sfc><scaling x="1" y="1" /></sfc></coordinateInfo></contentHeader>
  <types><dataTypes /><pous><pou name="rowchain" pouType="program">
    <interface><inputVars><variable name="x"><type><BOOL /></type></variable></inputVars><outputVars><variable name="y1"><type><BOOL /></type></variable><variable name="y2"><type><BOOL /></type></variable><variable name="y3"><type><BOOL /></type></variable></outputVars></interface>
    <body><LD>
      <leftPowerRail localId="0"><position x="0" y="0" /><connectionPointOut formalParameter="none" /></leftPowerRail>
      <contact localId="1" negated="false" storage="none" edge="none"><position x="20" y="10" /><connectionPointIn><connection refLocalId="0" /></connectionPointIn><connectionPointOut /><variable>x</variable></contact>
<coil localId="10" negated="false" storage="none" edge="none"><position x="100" y="0" /><connectionPointIn><connection refLocalId="1" /></connectionPointIn><connectionPointOut /><variable>y1</variable></coil>
<coil localId="11" negated="false" storage="none" edge="none"><position x="50" y="8" /><connectionPointIn><connection refLocalId="1" /></connectionPointIn><connectionPointOut /><variable>y2</variable></coil>
<coil localId="12" negated="false" storage="none" edge="none"><position x="10" y="16" /><connectionPointIn><connection refLocalId="1" /></connectionPointIn><connectionPointOut /><variable>y3</variable></coil>
      <rightPowerRail localId="2147483646"><position x="400" y="0" /><connectionPointIn /></rightPowerRail>
    </LD></body>
  </pou></pous></types>
  <instances><configurations><configuration name="Config0"><resource name="Res0"><task name="tcyclic" interval="T#10ms" priority="0"><pouInstance name="inst0" typeName="rowchain" /></task></resource></configuration></configurations></instances>
</project>
