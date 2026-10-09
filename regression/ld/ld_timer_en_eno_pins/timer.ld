<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-05"/>
  <contentHeader name="Timer"/>
  <types/><instances/>
  <pous>
    <pou name="Timer" pouType="program">
      <interface>
        <inputVars><variable name="IN1"><type><BOOL/></type></variable></inputVars>
        <localVars>
          <variable name="Q1"><type><BOOL/></type></variable>
          <variable name="ET1"><type><INT/></type></variable>
          <variable name="ENO1"><type><BOOL/></type></variable>
          <variable name="PT1"><type><INT/></type><initialValue><simpleValue value="2"/></initialValue></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="TON" instanceName="t1"><variable formalParameter="EN">IN1</variable><variable formalParameter="IN">IN1</variable><variable formalParameter="PT">PT1</variable><variable formalParameter="Q">Q1</variable><variable formalParameter="ET">ET1</variable><variable formalParameter="ENO">ENO1</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
