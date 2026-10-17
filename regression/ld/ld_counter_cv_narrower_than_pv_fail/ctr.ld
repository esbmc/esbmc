<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-06"/>
  <contentHeader name="T"/>
  <types/><instances/>
  <pous>
    <pou name="T" pouType="program">
      <interface>
        <inputVars>
          <variable name="Up"><type><BOOL/></type></variable>
        </inputVars>
        <localVars>
          <variable name="PV"><type><DINT/></type><initialValue><simpleValue value="40000"/></initialValue></variable>
          <variable name="CV"><type><INT/></type></variable>
          <variable name="Q"><type><BOOL/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="CTU" instanceName="c1"><variable formalParameter="CU">Up</variable><variable formalParameter="PV">PV</variable><variable formalParameter="Q">Q</variable><variable formalParameter="CV">CV</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
