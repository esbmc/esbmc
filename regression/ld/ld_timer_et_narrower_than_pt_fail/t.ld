<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-06"/>
  <contentHeader name="T"/>
  <types/><instances/>
  <pous>
    <pou name="T" pouType="program">
      <interface>
        <localVars>
          <variable name="In"><type><BOOL/></type></variable>
          <variable name="PT"><type><DINT/></type><initialValue><simpleValue value="40000"/></initialValue></variable>
          <variable name="ET"><type><INT/></type></variable>
          <variable name="Q"><type><BOOL/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="TON" instanceName="t1"><variable formalParameter="IN">In</variable><variable formalParameter="PT">PT</variable><variable formalParameter="Q">Q</variable><variable formalParameter="ET">ET</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
