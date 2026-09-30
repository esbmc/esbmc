<?xml version="1.0" encoding="utf-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productName="Probe" productVersion="1" creationDateTime="2026-09-28T00:00:00" />
  <contentHeader name="and_flag" />
  <!-- #7389: an AND block on a data path writes flag through an outVariable. -->
  <types><dataTypes /><pous><pou name="main" pouType="program">
    <interface>
      <inputVars>
        <variable name="a"><type><BOOL /></type></variable>
        <variable name="b"><type><BOOL /></type></variable>
      </inputVars>
      
      <outputVars><variable name="flag"><type><BOOL /></type></variable><variable name="q"><type><BOOL /></type></variable></outputVars>
    </interface>
    <body><LD>
      <leftPowerRail localId="0"><position x="0" y="0" /><connectionPointOut formalParameter="none" /></leftPowerRail>
      <inVariable localId="11"><position x="20" y="40" /><connectionPointOut /><expression>a</expression></inVariable>
      <inVariable localId="12"><position x="20" y="70" /><connectionPointOut /><expression>b</expression></inVariable>
      <block localId="13" typeName="AND">
        <position x="120" y="40" />
        <inputVariables>
          <variable formalParameter="IN1"><connectionPointIn><connection refLocalId="11" /></connectionPointIn></variable>
          <variable formalParameter="IN2"><connectionPointIn><connection refLocalId="12" /></connectionPointIn></variable>
        </inputVariables>
        <inOutVariables />
        <outputVariables><variable formalParameter="OUT"><connectionPointOut /></variable></outputVariables>
      </block>
      <outVariable localId="14"><position x="220" y="40" /><connectionPointIn><connection refLocalId="13" formalParameter="OUT" /></connectionPointIn><expression>flag</expression></outVariable>
      <contact localId="20" negated="false" storage="none" edge="none">
        <position x="20" y="120" />
        <connectionPointIn><connection refLocalId="0" /></connectionPointIn>
        <connectionPointOut /><variable>flag</variable>
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
