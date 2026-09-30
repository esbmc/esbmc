<?xml version="1.0" encoding="utf-8"?>
<!-- CTU0 whose R pin is driven by the Q output of CTU1, a counter with no R pin -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="Anonymous" productName="Benchmark Suite" productVersion="1.0" creationDateTime="2026-09-27T00:00:00" />
  <contentHeader name="g_ctu_reset_block"><coordinateInfo><fbd><scaling x="1" y="1" /></fbd><ld><scaling x="1" y="1" /></ld><sfc><scaling x="1" y="1" /></sfc></coordinateInfo></contentHeader>
  <types><dataTypes /><pous><pou name="g_ctu_reset_block" pouType="program">
    <interface><inputVars><variable name="Pulse"><type><BOOL /></type></variable><variable name="Stop"><type><BOOL /></type></variable></inputVars><outputVars><variable name="Done"><type><BOOL /></type></variable></outputVars></interface>
    <body><LD>
      <leftPowerRail localId="0"><position x="0" y="0" /><connectionPointOut formalParameter="none" /></leftPowerRail>
      <contact localId="3" negated="false" storage="none" edge="none"><position x="20" y="10" /><connectionPointIn><connection refLocalId="0" /></connectionPointIn><connectionPointOut /><variable>Pulse</variable></contact>
      <inVariable localId="4" height="20" width="60"><position x="20" y="60" /><connectionPointOut><relPosition x="60" y="10" /></connectionPointOut><expression>2</expression></inVariable>
      <contact localId="9" negated="false" storage="none" edge="none"><position x="20" y="120" /><connectionPointIn><connection refLocalId="0" /></connectionPointIn><connectionPointOut /><variable>Stop</variable></contact>
      <inVariable localId="10" height="20" width="60"><position x="20" y="150" /><connectionPointOut><relPosition x="60" y="10" /></connectionPointOut><expression>1</expression></inVariable>
      <block localId="11" width="80" height="80" typeName="CTU" instanceName="CTU1"><position x="60" y="120" /><inputVariables><variable formalParameter="CU"><connectionPointIn><relPosition x="0" y="20" /><connection refLocalId="9" /></connectionPointIn></variable><variable formalParameter="PV"><connectionPointIn><relPosition x="0" y="40" /><connection refLocalId="10" /></connectionPointIn></variable></inputVariables><inOutVariables /><outputVariables><variable formalParameter="Q"><connectionPointOut><relPosition x="80" y="20" /></connectionPointOut></variable><variable formalParameter="CV"><connectionPointOut><relPosition x="80" y="40" /></connectionPointOut></variable></outputVariables></block>
      <block localId="5" width="80" height="80" typeName="CTU" instanceName="CTU0"><position x="60" y="10" /><inputVariables><variable formalParameter="CU"><connectionPointIn><relPosition x="0" y="20" /><connection refLocalId="3" /></connectionPointIn></variable><variable formalParameter="PV"><connectionPointIn><relPosition x="0" y="40" /><connection refLocalId="4" /></connectionPointIn></variable><variable formalParameter="R"><connectionPointIn><relPosition x="0" y="60" /><connection refLocalId="11" formalParameter="Q" /></connectionPointIn></variable></inputVariables><inOutVariables /><outputVariables><variable formalParameter="Q"><connectionPointOut><relPosition x="80" y="20" /></connectionPointOut></variable><variable formalParameter="CV"><connectionPointOut><relPosition x="80" y="40" /></connectionPointOut></variable></outputVariables></block>
      <coil localId="6" negated="false" storage="none"><position x="160" y="10" /><connectionPointIn><connection refLocalId="5" /></connectionPointIn><connectionPointOut /><variable>Done</variable></coil>
      <rightPowerRail localId="2147483646"><position x="400" y="0" /><connectionPointIn><connection refLocalId="6" /></connectionPointIn></rightPowerRail>
    </LD></body>
  </pou></pous></types>
  <instances><configurations><configuration name="Config0"><resource name="Res0"><task name="tcyclic" interval="T#10ms" priority="0"><pouInstance name="inst0" typeName="g_ctu_reset_block" /></task></resource></configuration></configurations></instances>
</project>
