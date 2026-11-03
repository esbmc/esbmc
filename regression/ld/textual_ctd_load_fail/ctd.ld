<?xml version="1.0" encoding="UTF-8"?>
<!-- #7353: textual CTD whose LD input restores the preset (PV = 2). -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-27"/>
  <contentHeader name="CounterLoad"/>
  <types/><instances/>
  <pous>
    <pou name="CounterLoad" pouType="program">
      <interface>
        <inputVars>
          <variable name="Down"><type><BOOL/></type></variable>
          <variable name="Load"><type><BOOL/></type></variable>
        </inputVars>
        <localVars>
          <variable name="PV"><type><INT/></type><initialValue><simpleValue value="2"/></initialValue></variable>
          <variable name="CVd"><type><INT/></type><initialValue><simpleValue value="0"/></initialValue></variable>
        </localVars>
        <outputVars>
          <variable name="Empty"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="CTD" instanceName="cd1"><variable formalParameter="CD">Down</variable><variable formalParameter="LD">Load</variable><variable formalParameter="PV">PV</variable><variable formalParameter="Q">Empty</variable><variable formalParameter="CV">CVd</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
