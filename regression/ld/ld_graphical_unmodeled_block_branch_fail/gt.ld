<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://www.plcopen.org/xml/tc6_0201">
  <fileHeader companyName="x" productName="x" productVersion="1" creationDateTime="2026-10-11T00:00:00"/>
  <contentHeader name="gt"><coordinateInfo><ld><scaling x="10" y="10"/></ld></coordinateInfo></contentHeader>
  <types><dataTypes/><pous><pou name="gt" pouType="program">
    <interface>
      <inputVars><variable name="a"><type><BOOL/></type></variable><variable name="n"><type><INT/></type></variable></inputVars>
      <outputVars><variable name="y"><type><BOOL/></type></variable></outputVars>
    </interface>
    <body><LD>
      <leftPowerRail localId="0"><position x="0" y="0"/><connectionPointOut formalParameter="none"/></leftPowerRail>
      <contact localId="3" negated="false" storage="none" edge="none"><position x="20" y="10"/><connectionPointIn><connection refLocalId="0"/></connectionPointIn><connectionPointOut/><variable>a</variable></contact>
      <inVariable localId="5" height="20" width="60"><position x="20" y="60"/><connectionPointOut><relPosition x="60" y="10"/></connectionPointOut><expression>n</expression></inVariable>
      <inVariable localId="6" height="20" width="60"><position x="20" y="90"/><connectionPointOut><relPosition x="60" y="10"/></connectionPointOut><expression>3</expression></inVariable>
      <block localId="7" width="80" height="60" typeName="GT" instanceName="G0"><position x="60" y="60"/><inputVariables><variable formalParameter="IN1"><connectionPointIn><relPosition x="0" y="20"/><connection refLocalId="5"/></connectionPointIn></variable><variable formalParameter="IN2"><connectionPointIn><relPosition x="0" y="40"/><connection refLocalId="6"/></connectionPointIn></variable></inputVariables><inOutVariables/><outputVariables><variable formalParameter="OUT"><connectionPointOut><relPosition x="80" y="20"/></connectionPointOut></variable></outputVariables></block>
      <coil localId="8" negated="false" storage="none"><position x="160" y="10"/><connectionPointIn><connection refLocalId="3"/><connection refLocalId="7"/></connectionPointIn><connectionPointOut/><variable>y</variable></coil>
      <rightPowerRail localId="2147483646"><position x="400" y="0"/><connectionPointIn><connection refLocalId="8"/></connectionPointIn></rightPowerRail>
    </LD></body>
  </pou></pous></types>
  <instances><configurations><configuration name="c"><resource name="r"><task name="t" interval="T#10ms" priority="0"><pouInstance name="p" typeName="gt"/></task></resource></configuration></configurations></instances>
</project>
