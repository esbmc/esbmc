<?xml version="1.0" encoding="utf-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productName="Probe" productVersion="1" creationDateTime="2026-09-28T00:00:00" />
  <contentHeader name="probe" />
  <!-- #7389: the preset of T0 is computed by an ADD block. -->
  <types><dataTypes /><pous><pou name="main" pouType="program">
    <interface>
      <inputVars><variable name="a"><type><BOOL /></type></variable><variable name="n"><type><INT /></type></variable></inputVars>
      <outputVars><variable name="x"><type><BOOL /></type></variable><variable name="q"><type><BOOL /></type></variable></outputVars>
    </interface>
    <body><LD>
      <leftPowerRail localId="0"><position x="0" y="0" /><connectionPointOut formalParameter="none" /></leftPowerRail>
      <contact localId="10" negated="false" storage="none" edge="none"><position x="20" y="10" />
        <connectionPointIn><connection refLocalId="0" /></connectionPointIn><connectionPointOut /><variable>a</variable></contact>
      <inVariable localId="11"><position x="20" y="40" /><connectionPointOut /><expression>n</expression></inVariable>
      <inVariable localId="12"><position x="20" y="70" /><connectionPointOut /><expression>5</expression></inVariable>
      <block localId="15" typeName="ADD"><position x="80" y="40" />
        <inputVariables>
          <variable formalParameter="IN1"><connectionPointIn><connection refLocalId="11" /></connectionPointIn></variable>
          <variable formalParameter="IN2"><connectionPointIn><connection refLocalId="12" /></connectionPointIn></variable>
        </inputVariables><inOutVariables />
        <outputVariables><variable formalParameter="OUT"><connectionPointOut /></variable></outputVariables>
      </block>
      <block localId="13" typeName="TON" instanceName="T0"><position x="120" y="10" />
        <inputVariables>
          <variable formalParameter="IN"><connectionPointIn><connection refLocalId="10" /></connectionPointIn></variable>
          <variable formalParameter="PT"><connectionPointIn><connection refLocalId="15" formalParameter="OUT" /></connectionPointIn></variable>
        </inputVariables><inOutVariables />
        <outputVariables><variable formalParameter="Q"><connectionPointOut /></variable><variable formalParameter="ET"><connectionPointOut /></variable></outputVariables>
      </block>
      <coil localId="16" negated="false" storage="none"><position x="200" y="10" />
        <connectionPointIn><connection refLocalId="13" formalParameter="Q" /></connectionPointIn><connectionPointOut /><variable>x</variable></coil>
      <contact localId="20" negated="false" storage="none" edge="none"><position x="20" y="120" />
        <connectionPointIn><connection refLocalId="0" /></connectionPointIn><connectionPointOut /><variable>x</variable></contact>
      <coil localId="21" negated="false" storage="none"><position x="60" y="120" />
        <connectionPointIn><connection refLocalId="20" /></connectionPointIn><connectionPointOut /><variable>q</variable></coil>
      <rightPowerRail localId="2147483646"><position x="300" y="0" /><connectionPointIn /></rightPowerRail>
    </LD></body>
  </pou></pous></types>
  <instances><configurations><configuration name="C"><resource name="R"><task name="t" interval="T#10ms" priority="0"><pouInstance name="i" typeName="main" /></task></resource></configuration></configurations></instances>
</project>
