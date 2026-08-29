<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-07-30"/>
  <contentHeader name="CounterSaturate"/>
  <types/><instances/>
  <pous>
    <pou name="CounterSaturate" pouType="program">
      <interface>
        <inputVars>
          <variable name="Up"><type><BOOL/></type></variable>
        </inputVars>
        <localVars>
          <variable name="PV"><type><INT/></type><initialValue><simpleValue value="2"/></initialValue></variable>
          <variable name="CVu"><type><INT/></type><initialValue><simpleValue value="0"/></initialValue></variable>
        </localVars>
        <outputVars>
          <variable name="CUQ"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="CTU" instanceName="cu1"><variable formalParameter="CU">Up</variable><variable formalParameter="PV">PV</variable><variable formalParameter="Q">CUQ</variable><variable formalParameter="CV">CVu</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
