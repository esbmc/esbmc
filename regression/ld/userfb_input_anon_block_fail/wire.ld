<?xml version="1.0" encoding="UTF-8"?>
<!-- #7580: a user FB input wired to the Q pin of an anonymous CTU, which the
     graphical resolver names blk<localId>. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-27"/>
  <contentHeader name="WIRE"/>
  <types/><instances/>
  <pous>
    <pou name="PASS" pouType="functionBlock">
      <interface>
        <inputVars><variable name="A"><type><BOOL/></type></variable></inputVars>
        <outputVars><variable name="Q"><type><BOOL/></type></variable></outputVars>
      </interface>
      <body><ST><![CDATA[
Q := A;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface>
        <inputVars><variable name="X"><type><BOOL/></type></variable></inputVars>
        <outputVars>
          <variable name="Done"><type><BOOL/></type></variable>
          <variable name="Y"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body><LD>
        <leftPowerRail localId="0"><connectionPointOut formalParameter="none"/></leftPowerRail>
        <contact localId="3" negated="false" edge="none"><connectionPointIn><connection refLocalId="0"/></connectionPointIn><variable>X</variable></contact>
        <inVariable localId="4"><expression>2</expression></inVariable>
        <block localId="5" typeName="CTU">
          <inputVariables>
            <variable formalParameter="CU"><connectionPointIn><connection refLocalId="3"/></connectionPointIn></variable>
            <variable formalParameter="PV"><connectionPointIn><connection refLocalId="4"/></connectionPointIn></variable>
          </inputVariables>
        </block>
        <coil localId="6" negated="false" storage="none"><connectionPointIn><connection refLocalId="5"/></connectionPointIn><variable>Done</variable></coil>
        <rightPowerRail localId="7"><connectionPointIn><connection refLocalId="6"/></connectionPointIn></rightPowerRail>
        <block localId="10" typeName="PASS" instanceName="p0">
          <inputVariables>
            <variable formalParameter="A"><connectionPointIn><connection refLocalId="5" formalParameter="Q"/></connectionPointIn></variable>
          </inputVariables>
        </block>
        <outVariable localId="20"><expression>Y</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="Q"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
