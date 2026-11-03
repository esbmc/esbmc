<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-04"/>
  <contentHeader name="Timer"/>
  <types/><instances/>
  <pous>
    <pou name="Timer" pouType="program">
      <interface>
        <inputVars><variable name="IN1"><type><BOOL/></type></variable></inputVars>
        <localVars>
          <variable name="H1"><type><BOOL/></type></variable>
          <variable name="H2"><type><BOOL/></type></variable>
          <variable name="Q1"><type><BOOL/></type></variable>
          <variable name="ET1"><type><INT/></type></variable>
          <variable name="PT1"><type><INT/></type><initialValue><simpleValue value="0"/></initialValue></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="2"><contact variable="H1"/><coil variable="H2"/></rung>
          <rung localId="3"><contact variable="IN1"/><coil variable="H1"/></rung>
          <rung localId="4"><block typeName="TOF" instanceName="t1"><variable formalParameter="IN">IN1</variable><variable formalParameter="PT">PT1</variable><variable formalParameter="Q">Q1</variable><variable formalParameter="ET">ET1</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
