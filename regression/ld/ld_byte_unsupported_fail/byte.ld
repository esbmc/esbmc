<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-06"/>
  <contentHeader name="Wrap"/>
  <types/><instances/>
  <pous>
    <pou name="Wrap" pouType="program">
      <interface>
        <localVars>
          <variable name="K"><type><INT/></type><initialValue><simpleValue value="256"/></initialValue></variable>
          <variable name="P"><type><INT/></type></variable>
          <variable name="NonZero"><type><BOOL/></type></variable>
          <variable name="Bt"><type><BYTE/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="MUL" instanceName="m1"><variable formalParameter="IN1">K</variable><variable formalParameter="IN2">K</variable><variable formalParameter="OUT">P</variable></block></rung>
          <rung localId="2"><contact variable="Bt"/><coil variable="NonZero"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
