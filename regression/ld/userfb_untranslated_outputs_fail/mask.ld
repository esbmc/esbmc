<?xml version="1.0" encoding="UTF-8"?>
<!-- #7579: AND on an INT operand is bitwise in IEC 61131-3 and not modelled. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-27"/>
  <contentHeader name="MASK"/>
  <types/><instances/>
  <pous>
    <pou name="MASK" pouType="functionBlock">
      <interface>
        <inputVars>
          <variable name="N"><type><INT/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="OUT"><type><BOOL/></type></variable>
          <variable name="MASKED"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body><ST><![CDATA[
OUT := TRUE;
MASKED := N AND B;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface><outputVars><variable name="Q"><type><BOOL/></type></variable></outputVars></interface>
      <body><LD>
        <block localId="10" typeName="MASK" instanceName="m0"/>
        <outVariable localId="11"><expression>Q</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="OUT"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
