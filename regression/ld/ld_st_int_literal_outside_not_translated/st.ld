<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="X" productVersion="1.0" creationDateTime="2026-10-06"/>
  <contentHeader name="LOOP"/>
  <types/><instances/>
  <pous>
    <pou name="LOOP" pouType="functionBlock">
      <interface>
        <outputVars><variable name="OUT"><type><BOOL/></type></variable></outputVars>
        <localVars>
          <variable name="i"><type><INT/></type></variable>
          <variable name="d"><type><DINT/></type></variable>
          <variable name="x"/>
        </localVars>
      </interface>
      <body><ST><![CDATA[
i := 40000;
OUT := TRUE;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface><outputVars><variable name="Safe"><type><BOOL/></type></variable></outputVars></interface>
      <body><LD>
        <block localId="10" typeName="LOOP" instanceName="c0"/>
        <outVariable localId="11"><expression>Safe</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="OUT"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
