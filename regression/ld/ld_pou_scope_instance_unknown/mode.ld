<?xml version="1.0" encoding="utf-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="SAFE-LD" productName="Probe" productVersion="1" creationDateTime="2026-09-29T00:00:00" />
  <contentHeader name="probe" />
  <!-- #7581: the program instantiates PASS as p0 and wires p0's input mode from feed;
       p0__mode and the program's own mode, which starts FALSE, are distinct. -->
  <types><dataTypes /><pous>
    <pou name="PASS" pouType="functionBlock">
      <interface>
        <inputVars><variable name="mode"><type><BOOL /></type></variable></inputVars>
        <outputVars><variable name="OUT"><type><BOOL /></type></variable></outputVars>
      </interface>
      <body><ST><![CDATA[
OUT := mode;
]]></ST></body>
    </pou>
    <pou name="main" pouType="program">
      <interface>
        <inputVars><variable name="feed"><type><BOOL /></type></variable></inputVars>
        <localVars><variable name="mode"><type><BOOL /></type></variable><variable name="p0"><type><derived name="PASS" /></type></variable></localVars>
        <outputVars><variable name="q"><type><BOOL /></type></variable><variable name="seen"><type><BOOL /></type></variable></outputVars>
      </interface>
      <body><LD>
        <leftPowerRail localId="0"><position x="0" y="0" /><connectionPointOut formalParameter="none" /></leftPowerRail>
        <contact localId="10" negated="false" storage="none" edge="none"><position x="20" y="10" />
          <connectionPointIn><connection refLocalId="0" /></connectionPointIn><connectionPointOut /><variable>mode</variable></contact>
        <coil localId="11" negated="false" storage="none"><position x="60" y="10" />
          <connectionPointIn><connection refLocalId="10" /></connectionPointIn><connectionPointOut /><variable>q</variable></coil>
        <contact localId="20" negated="false" storage="none" edge="none"><position x="20" y="60" />
          <connectionPointIn><connection refLocalId="0" /></connectionPointIn><connectionPointOut /><variable>q</variable></contact>
        <coil localId="21" negated="false" storage="none"><position x="60" y="60" />
          <connectionPointIn><connection refLocalId="20" /></connectionPointIn><connectionPointOut /><variable>mode</variable></coil>
        <inVariable localId="30"><position x="20" y="120" /><connectionPointOut><relPosition x="40" y="10" /></connectionPointOut><expression>feed</expression></inVariable>
        <block localId="31" typeName="PASS" instanceName="p0"><position x="100" y="110" />
          <inputVariables><variable formalParameter="mode"><connectionPointIn><relPosition x="0" y="20" /><connection refLocalId="30" /></connectionPointIn></variable></inputVariables>
          <inOutVariables />
          <outputVariables><variable formalParameter="OUT"><connectionPointOut><relPosition x="60" y="20" /></connectionPointOut></variable></outputVariables>
        </block>
        <outVariable localId="32"><position x="200" y="120" /><connectionPointIn><connection refLocalId="31" formalParameter="OUT"><position x="160" y="130" /></connection></connectionPointIn><expression>seen</expression></outVariable>
        <rightPowerRail localId="2147483646"><position x="300" y="0" /><connectionPointIn /></rightPowerRail>
      </LD></body>
    </pou>
  </pous></types>
  <instances><configurations><configuration name="C"><resource name="R"><task name="t" interval="T#10ms" priority="0"><pouInstance name="i" typeName="main" /></task></resource></configuration></configurations></instances>
</project>
