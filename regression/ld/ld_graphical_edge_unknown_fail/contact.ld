<?xml version="1.0" encoding="utf-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="ESBMC" productName="probe" productVersion="1.0" creationDateTime="2026-10-07T00:00:00" />
  <contentHeader name="edge1"><coordinateInfo><fbd><scaling x="1" y="1" /></fbd><ld><scaling x="1" y="1" /></ld><sfc><scaling x="1" y="1" /></sfc></coordinateInfo></contentHeader>
  <types><dataTypes /><pous><pou name="edge1" pouType="program">
    <interface><inputVars><variable name="x"><type><BOOL /></type></variable></inputVars><outputVars><variable name="y"><type><BOOL /></type></variable></outputVars></interface>
    <body><LD>
      <leftPowerRail localId="0"><position x="0" y="0" /><connectionPointOut formalParameter="none" /></leftPowerRail>
      <contact localId="1" negated="false" storage="none" edge="Rising"><position x="20" y="10" /><connectionPointIn><connection refLocalId="0" /></connectionPointIn><connectionPointOut /><variable>x</variable></contact>
<coil localId="2" negated="false" storage="none" edge="none"><position x="100" y="10" /><connectionPointIn><connection refLocalId="1" /></connectionPointIn><connectionPointOut /><variable>y</variable></coil>
      <rightPowerRail localId="2147483646"><position x="400" y="0" /><connectionPointIn /></rightPowerRail>
    </LD></body>
  </pou></pous></types>
  <instances><configurations><configuration name="Config0"><resource name="Res0"><task name="tcyclic" interval="T#10ms" priority="0"><pouInstance name="inst0" typeName="edge1" /></task></resource></configuration></configurations></instances>
</project>
