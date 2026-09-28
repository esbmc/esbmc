<?xml version="1.0" encoding="utf-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="SAFE-LD" productName="Probe" productVersion="1" creationDateTime="2026-09-28T00:00:00" />
  <contentHeader name="probe" />
  <!-- Under --ld-closed-world only the program writes the %M variable mode. -->
  <types><dataTypes /><pous><pou name="main" pouType="program">
    <interface>
      <outputVars><variable name="q"><type><BOOL /></type></variable></outputVars>
      <localVars><variable name="mode" address="%MX0.0"><type><BOOL /></type></variable></localVars>
    </interface>
    <body><LD>
      <leftPowerRail localId="0"><position x="0" y="0" /><connectionPointOut formalParameter="none" /></leftPowerRail>
      <contact localId="10" negated="false" storage="none" edge="none"><position x="20" y="10" />
        <connectionPointIn><connection refLocalId="0" /></connectionPointIn><connectionPointOut /><variable>mode</variable></contact>
      <coil localId="11" negated="false" storage="none"><position x="60" y="10" />
        <connectionPointIn><connection refLocalId="10" /></connectionPointIn><connectionPointOut /><variable>q</variable></coil>
      <coil localId="21" negated="false" storage="reset"><position x="60" y="60" />
        <connectionPointIn><connection refLocalId="0" /></connectionPointIn><connectionPointOut /><variable>mode</variable></coil>
      <rightPowerRail localId="2147483646"><position x="300" y="0" /><connectionPointIn /></rightPowerRail>
    </LD></body>
  </pou></pous></types>
  <instances><configurations><configuration name="C"><resource name="R"><task name="t" interval="T#10ms" priority="0"><pouInstance name="i" typeName="main" /></task></resource></configuration></configurations></instances>
</project>
