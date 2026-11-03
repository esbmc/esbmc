<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-06"/>
  <contentHeader name="T"/>
  <types/><instances/>
  <pous>
    <pou name="T" pouType="program">
      <interface>
        <localVars>
          <variable name="K"><type><INT/></type><initialValue><simpleValue value="4096"/></initialValue></variable>
          <variable name="P1"><type><INT/></type></variable>
          <variable name="P"><type><INT/></type></variable>
          <variable name="NonZero"><type><BOOL/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="ADD" instanceName="a1"><variable formalParameter="IN1">K</variable><variable formalParameter="IN2">K</variable><variable formalParameter="OUT">P1</variable></block><block typeName="ADD" instanceName="a2"><variable formalParameter="IN1">P1</variable><variable formalParameter="IN2">P1</variable><variable formalParameter="OUT">P</variable></block></rung>
          <rung localId="2"><contact variable="P"/><coil variable="NonZero"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
