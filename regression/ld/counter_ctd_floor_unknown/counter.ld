<?xml version="1.0" encoding="UTF-8"?>
<!-- CTD stops counting down at 0, as in MATIEC: D = CV + 1 stays positive, so a CTU with preset D and no count keeps Full off. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-29"/>
  <contentHeader name="CtdFloor"/>
  <types/><instances/>
  <pous>
    <pou name="CtdFloor" pouType="program">
      <interface>
        <inputVars><variable name="Down"><type><BOOL/></type></variable></inputVars>
        <localVars><variable name="Never"><type><BOOL/></type></variable><variable name="One"><type><INT/></type><initialValue><simpleValue value="1"/></initialValue></variable><variable name="P"><type><INT/></type><initialValue><simpleValue value="0"/></initialValue></variable><variable name="CVd"><type><INT/></type><initialValue><simpleValue value="0"/></initialValue></variable><variable name="D"><type><INT/></type><initialValue><simpleValue value="1"/></initialValue></variable><variable name="CV2"><type><INT/></type><initialValue><simpleValue value="0"/></initialValue></variable><variable name="Empty"><type><BOOL/></type></variable><variable name="Full"><type><BOOL/></type></variable></localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="CTD" instanceName="cd1"><variable formalParameter="CD">Down</variable><variable formalParameter="PV">P</variable><variable formalParameter="Q">Empty</variable><variable formalParameter="CV">CVd</variable></block></rung>
          <rung localId="2"><block typeName="ADD" instanceName="a1"><variable formalParameter="IN1">CVd</variable><variable formalParameter="IN2">One</variable><variable formalParameter="OUT">D</variable></block></rung>
          <rung localId="3"><block typeName="CTU" instanceName="cu2"><variable formalParameter="CU">Never</variable><variable formalParameter="R">Never</variable><variable formalParameter="PV">D</variable><variable formalParameter="Q">Full</variable><variable formalParameter="CV">CV2</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
