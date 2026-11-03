<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-06"/>
  <contentHeader name="Mixed"/>
  <types/><instances/>
  <pous>
    <pou name="Mixed" pouType="program">
      <interface>
        <localVars>
          <variable name="Small"><type><INT/></type><initialValue><simpleValue value="-3"/></initialValue></variable>
          <variable name="Large"><type><DINT/></type><initialValue><simpleValue value="100000"/></initialValue></variable>
          <variable name="Sum"><type><DINT/></type></variable>
          <variable name="Copy"><type><DINT/></type></variable>
          <variable name="Ok"><type><BOOL/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="ADD" instanceName="a1"><variable formalParameter="IN1">Small</variable><variable formalParameter="IN2">Large</variable><variable formalParameter="OUT">Sum</variable></block></rung>
          <rung localId="2"><block typeName="MOVE" instanceName="mv1"><variable formalParameter="IN1">Small</variable><variable formalParameter="OUT">Copy</variable></block></rung>
          <rung localId="3"><contact variable="Sum"/><contact variable="Copy"/><coil variable="Ok"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
