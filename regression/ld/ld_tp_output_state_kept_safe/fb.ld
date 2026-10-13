<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-11"/>
  <contentHeader name="P"/>
  <types/><instances/>
  <pous>
    <pou name="P" pouType="program">
      <interface>
        <inputVars><variable name="IN1"><type><BOOL/></type></variable></inputVars>
        <outputVars><variable name="Seen"><type><BOOL/></type></variable></outputVars>
        <localVars><variable name="H1"><type><BOOL/></type></variable><variable name="H2"><type><BOOL/></type></variable><variable name="H3"><type><BOOL/></type></variable><variable name="PT1"><type><INT/></type><initialValue><simpleValue value="2"/></initialValue></variable><variable name="ET1"><type><INT/></type></variable><variable name="Q1"><type><BOOL/></type></variable></localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="H2"/><coil variable="H3"/></rung>
          <rung localId="2"><contact variable="H1"/><coil variable="H2"/></rung>
          <rung localId="3"><contact variable="IN1"/><coil variable="H1"/></rung>
          <rung localId="4"><block typeName="TP" instanceName="p1"><variable formalParameter="IN">IN1</variable><variable formalParameter="PT">PT1</variable><variable formalParameter="Q">Q1</variable><variable formalParameter="ET">ET1</variable></block></rung>
          <rung localId="5"><contact variable="Q1"/><coil variable="Seen"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
