<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-04"/>
  <contentHeader name="Counters"/>
  <types/><instances/>
  <pous>
    <pou name="Counters" pouType="program">
      <interface>
        <inputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </inputVars>
        <localVars>
          <variable name="SEEN"><type><BOOL/></type></variable>
          <variable name="QA"><type><BOOL/></type></variable>
          <variable name="QB"><type><BOOL/></type></variable>
          <variable name="CVA"><type><INT/></type></variable>
          <variable name="CVB"><type><INT/></type></variable>
          <variable name="PV1"><type><INT/></type><initialValue><simpleValue value="1"/></initialValue></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="B"/><coil kind="set" variable="SEEN"/></rung>
          <rung localId="2"><block typeName="CTU"><variable formalParameter="CU">A</variable><variable formalParameter="PV">PV1</variable><variable formalParameter="Q">QA</variable><variable formalParameter="CV">CVA</variable></block></rung>
          <rung localId="3"><block typeName="CTU"><variable formalParameter="CU">B</variable><variable formalParameter="PV">PV1</variable><variable formalParameter="Q">QB</variable><variable formalParameter="CV">CVB</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
