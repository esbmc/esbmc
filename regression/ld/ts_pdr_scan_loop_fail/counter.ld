<?xml version="1.0" encoding="UTF-8"?>
<!-- CTU stops counting up at PV, as in MATIEC: D = CV - PV stays at most 0, so a CTU with preset D and no count keeps Ok on. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-29"/>
  <contentHeader name="CtuCeiling"/>
  <types/><instances/>
  <pous>
    <pou name="CtuCeiling" pouType="program">
      <interface>
        <inputVars><variable name="Up"><type><BOOL/></type></variable></inputVars>
        <localVars><variable name="Never"><type><BOOL/></type></variable><variable name="P"><type><INT/></type><initialValue><simpleValue value="1"/></initialValue></variable><variable name="CVu"><type><INT/></type><initialValue><simpleValue value="0"/></initialValue></variable><variable name="D"><type><INT/></type><initialValue><simpleValue value="-1"/></initialValue></variable><variable name="CV2"><type><INT/></type><initialValue><simpleValue value="0"/></initialValue></variable><variable name="Done"><type><BOOL/></type></variable><variable name="Ok"><type><BOOL/></type><initialValue><simpleValue value="TRUE"/></initialValue></variable></localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="CTU" instanceName="cu1"><variable formalParameter="CU">Up</variable><variable formalParameter="R">Never</variable><variable formalParameter="PV">P</variable><variable formalParameter="Q">Done</variable><variable formalParameter="CV">CVu</variable></block></rung>
          <rung localId="2"><block typeName="SUB" instanceName="s1"><variable formalParameter="IN1">CVu</variable><variable formalParameter="IN2">P</variable><variable formalParameter="OUT">D</variable></block></rung>
          <rung localId="3"><block typeName="CTU" instanceName="cu2"><variable formalParameter="CU">Never</variable><variable formalParameter="R">Never</variable><variable formalParameter="PV">D</variable><variable formalParameter="Q">Ok</variable><variable formalParameter="CV">CV2</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
