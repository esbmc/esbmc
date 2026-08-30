<?xml version="1.0" encoding="UTF-8"?>
<!-- A function block's ST body is translated only once the definition
     registers; with no output pin this one was dropped and its output
     havocked instead, giving an unexplained counterexample (#7354). -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-08-27"/>
  <contentHeader name="NOPIN"/>
  <types/>
  <instances/>
  <pous>
    <pou name="NOPIN" pouType="functionBlock">
      <interface>
        <localVars><variable name="i"><type><INT/></type></variable></localVars>
      </interface>
      <body><ST><![CDATA[
i := 1;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface>
        <outputVars><variable name="Safe"><type><BOOL/></type></variable></outputVars>
      </interface>
      <body>
        <LD>
          <block localId="10" typeName="NOPIN" instanceName="n0"/>
          <outVariable localId="11"><expression>Safe</expression>
            <connectionPointIn><connection refLocalId="10" formalParameter="OUT"/></connectionPointIn></outVariable>
        </LD>
      </body>
    </pou>
  </pous>
</project>
