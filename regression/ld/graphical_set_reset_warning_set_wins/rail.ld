<?xml version="1.0" encoding="utf-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="SAFE-LD" productName="Probe" productVersion="1" creationDateTime="2026-09-28T00:00:00" />
  <!-- mode has a reset coil above a set coil: the set runs later and wins. -->
  <contentHeader name="rail"><coordinateInfo><fbd><scaling x="1" y="1" /></fbd><ld><scaling x="1" y="1" /></ld><sfc><scaling x="1" y="1" /></sfc></coordinateInfo></contentHeader>
  <types><dataTypes /><pous><pou name="main" pouType="program">
    <interface>
      <inputVars><variable name="a"><type><BOOL /></type></variable><variable name="b"><type><BOOL /></type></variable></inputVars>
      <localVars><variable name="mode"><type><BOOL /></type></variable></localVars>
    </interface>
    <body><LD>
      <leftPowerRail localId="1" height="100" width="3"><position x="10" y="10" />
        <connectionPointOut formalParameter=""><relPosition x="3" y="10" /></connectionPointOut>
        <connectionPointOut formalParameter=""><relPosition x="3" y="60" /></connectionPointOut></leftPowerRail>
      <contact localId="2" height="15" width="21" negated="false"><position x="50" y="62" />
        <connectionPointIn><relPosition x="0" y="8" /><connection refLocalId="1"><position x="50" y="70" /><position x="13" y="70" /></connection></connectionPointIn>
        <connectionPointOut><relPosition x="21" y="8" /></connectionPointOut><variable>a</variable></contact>
      <coil localId="3" height="15" width="21" negated="false" storage="set"><position x="150" y="62" />
        <connectionPointIn><relPosition x="0" y="8" /><connection refLocalId="2"><position x="150" y="70" /><position x="71" y="70" /></connection></connectionPointIn>
        <connectionPointOut><relPosition x="21" y="8" /></connectionPointOut><variable>mode</variable></coil>
      <contact localId="4" height="15" width="21" negated="false"><position x="50" y="12" />
        <connectionPointIn><relPosition x="0" y="8" /><connection refLocalId="1"><position x="50" y="20" /><position x="13" y="20" /></connection></connectionPointIn>
        <connectionPointOut><relPosition x="21" y="8" /></connectionPointOut><variable>b</variable></contact>
      <coil localId="5" height="15" width="21" negated="false" storage="reset"><position x="150" y="12" />
        <connectionPointIn><relPosition x="0" y="8" /><connection refLocalId="4"><position x="150" y="20" /><position x="71" y="20" /></connection></connectionPointIn>
        <connectionPointOut><relPosition x="21" y="8" /></connectionPointOut><variable>mode</variable></coil>
      <rightPowerRail localId="6" height="100" width="3"><position x="250" y="10" />
        <connectionPointIn><relPosition x="0" y="60" /><connection refLocalId="5"><position x="250" y="20" /><position x="171" y="20" /></connection></connectionPointIn>
        <connectionPointIn><relPosition x="0" y="10" /><connection refLocalId="3"><position x="250" y="70" /><position x="171" y="70" /></connection></connectionPointIn>
      </rightPowerRail>
    </LD></body>
  </pou></pous></types>
  <instances><configurations><configuration name="C"><resource name="R"><task name="t" interval="T#10ms" priority="0"><pouInstance name="i" typeName="main" /></task></resource></configuration></configurations></instances>
</project>
