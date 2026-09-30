<?xml version="1.0" encoding="utf-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productName="Probe" productVersion="1" creationDateTime="2026-09-28T00:00:00" />
  <contentHeader name="dangling_out" />
  <!-- An outVariable wired to nothing drives nothing, so it is not refused. -->
  <types><dataTypes /><pous><pou name="main" pouType="program">
    <interface>
      <inputVars>
        <variable name="a"><type><BOOL /></type></variable>
      </inputVars>
      
      <outputVars><variable name="flag"><type><BOOL /></type></variable><variable name="q"><type><BOOL /></type></variable></outputVars>
    </interface>
    <body><LD>
      <leftPowerRail localId="0"><position x="0" y="0" /><connectionPointOut formalParameter="none" /></leftPowerRail>
      <inVariable localId="11"><position x="20" y="40" /><connectionPointOut /><expression>a</expression></inVariable>
      <outVariable localId="14"><position x="220" y="40" /><connectionPointIn /><expression>flag</expression></outVariable>
      <contact localId="20" negated="false" storage="none" edge="none">
        <position x="20" y="120" />
        <connectionPointIn><connection refLocalId="0" /></connectionPointIn>
        <connectionPointOut /><variable>a</variable>
      </contact>
      <coil localId="21" negated="false" storage="none">
        <position x="60" y="120" />
        <connectionPointIn><connection refLocalId="20" /></connectionPointIn>
        <connectionPointOut /><variable>q</variable>
      </coil>
      <rightPowerRail localId="2147483646"><position x="300" y="0" /><connectionPointIn /></rightPowerRail>
    </LD></body>
  </pou></pous></types>
  <instances><configurations><configuration name="Config0"><resource name="Res0"><task name="t" interval="T#10ms" priority="0"><pouInstance name="inst0" typeName="main" /></task></resource></configuration></configurations></instances>
</project>
