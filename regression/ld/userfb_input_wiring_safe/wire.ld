<?xml version="1.0" encoding="UTF-8"?>
<!-- #7580: user FB inputs bound to what they are wired to: a program
     variable, another block's output pin, and BOOL and integer literals. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-27"/>
  <contentHeader name="WIRE"/>
  <types/><instances/>
  <pous>
    <pou name="PASS" pouType="functionBlock">
      <interface>
        <inputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="N"><type><INT/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="BIG"><type><BOOL/></type></variable>
          <variable name="Q"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body><ST><![CDATA[
Q := A;
BIG := N > 4;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface>
        <inputVars><variable name="X"><type><BOOL/></type></variable></inputVars>
        <outputVars>
          <variable name="Y"><type><BOOL/></type></variable>
          <variable name="Z"><type><BOOL/></type></variable>
          <variable name="W"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body><LD>
        <inVariable localId="1"><expression>X</expression></inVariable>
        <inVariable localId="2"><expression>5</expression></inVariable>
        <inVariable localId="3"><expression>-3</expression></inVariable>
        <inVariable localId="4"><expression>TRUE</expression></inVariable>
        <block localId="10" typeName="PASS" instanceName="p0">
          <inputVariables>
            <variable formalParameter="A"><connectionPointIn><connection refLocalId="1"/></connectionPointIn></variable>
            <variable formalParameter="N"><connectionPointIn><connection refLocalId="2"/></connectionPointIn></variable>
          </inputVariables>
        </block>
        <block localId="11" typeName="PASS" instanceName="p1">
          <inputVariables>
            <variable formalParameter="A"><connectionPointIn><connection refLocalId="10" formalParameter="Q"/></connectionPointIn></variable>
            <variable formalParameter="N"><connectionPointIn><connection refLocalId="3"/></connectionPointIn></variable>
          </inputVariables>
        </block>
        <block localId="12" typeName="PASS" instanceName="p2">
          <inputVariables>
            <variable formalParameter="A"><connectionPointIn><connection refLocalId="4"/></connectionPointIn></variable>
          </inputVariables>
        </block>
        <outVariable localId="20"><expression>Y</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="Q"/></connectionPointIn></outVariable>
        <outVariable localId="21"><expression>Z</expression>
          <connectionPointIn><connection refLocalId="11" formalParameter="Q"/></connectionPointIn></outVariable>
        <outVariable localId="22"><expression>W</expression>
          <connectionPointIn><connection refLocalId="12" formalParameter="Q"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
