<?xml version="1.0" encoding="UTF-8"?>
<!-- #7580: a user FB input fed by a contact is not modelled; it stays
     nondeterministic, and the front end says so. -->
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
        <leftPowerRail localId="1"><connectionPointOut formalParameter="none"/></leftPowerRail>
        <contact localId="2" negated="false"><connectionPointIn><connection refLocalId="1"/></connectionPointIn><connectionPointOut/><variable>X</variable></contact>
        <block localId="10" typeName="PASS" instanceName="p0">
          <inputVariables>
            <variable formalParameter="A"><connectionPointIn><connection refLocalId="2"/></connectionPointIn></variable>
          </inputVariables>
        </block>
        <outVariable localId="20"><expression>Y</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="Q"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
