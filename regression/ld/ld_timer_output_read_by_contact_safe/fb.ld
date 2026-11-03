<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-11"/>
  <contentHeader name="P"/>
  <types/><instances/>
  <pous>
    <pou name="P" pouType="program">
      <interface>
        <outputVars><variable name="Out"><type><BOOL/></type></variable></outputVars>
        <localVars><variable name="Never"><type><BOOL/></type></variable><variable name="One"><type><BOOL/></type></variable><variable name="PT1"><type><INT/></type><initialValue><simpleValue value="3"/></initialValue></variable><variable name="ET1"><type><INT/></type></variable><variable name="Q1"><type><BOOL/></type></variable></localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="Q1"/><coil variable="Out"/></rung>
          <rung localId="2"><block typeName="TON" instanceName="t1"><variable formalParameter="IN">Never</variable><variable formalParameter="PT">PT1</variable><variable formalParameter="Q">Q1</variable><variable formalParameter="ET">ET1</variable></block></rung>
          <rung localId="8"><coil variable="One"/></rung>
          <rung localId="9"><contact variable="One" negated="true"/><coil variable="Never"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
