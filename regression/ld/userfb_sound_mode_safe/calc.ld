<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-26"/>
  <contentHeader name="CALC"/>
  <types/><instances/>
  <pous>
    <pou name="CALC" pouType="functionBlock">
      <interface>
        <outputVars><variable name="OUT"><type><BOOL/></type></variable></outputVars>
      </interface>
      <body><ST><![CDATA[
OUT := mystery() = 1;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface><outputVars><variable name="Open"><type><BOOL/></type></variable></outputVars></interface>
      <body><LD>
        <block localId="10" typeName="CALC" instanceName="c0"/>
        <outVariable localId="11"><expression>Open</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="OUT"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
