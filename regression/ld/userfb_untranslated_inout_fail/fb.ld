<?xml version="1.0" encoding="UTF-8"?>
<!-- The INT operand of AND keeps MASK's body untranslated; Q reads its
     in/out pin IO, which the body sets. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-28"/>
  <contentHeader name="IOT"/>
  <types/><instances/>
  <pous>
    <pou name="MASK" pouType="functionBlock">
      <interface>
        <inputVars>
          <variable name="N"><type><INT/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </inputVars>
        <inOutVars><variable name="IO"><type><BOOL/></type></variable></inOutVars>
        <outputVars>
          <variable name="MASKED"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body><ST><![CDATA[
IO := TRUE;
MASKED := N AND B;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface><outputVars><variable name="Q"><type><BOOL/></type></variable></outputVars></interface>
      <body><LD>
        <block localId="10" typeName="MASK" instanceName="m0"/>
        <outVariable localId="11"><expression>Q</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="IO"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
